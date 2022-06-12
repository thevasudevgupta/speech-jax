import dataclasses
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from pydantic import BaseModel
from transformers import (FlaxWav2Vec2ForCTC, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor)
from transformers.models.wav2vec2.modeling_flax_wav2vec2 import \
    _compute_mask_indices

from speech_jax import training
from speech_jax.training import (TrainerConfig, TrainingStepOutput,
                                 ValidationStepOutput)
from speech_jax.tx_utils import create_tx
from speech_jax.utils import read_yaml
from speech_jax.hf_utils import hf_save_fn
import optax
from speech_jax.tx_utils import linear_scheduler_with_warmup
from datasets import interleave_datasets, load_dataset
import sys

# python3 projects/finetune_wav2vec2.py "projects/configs/wav2vec2_asr"

print(jax.devices())
configs = read_yaml(sys.args[1])
print(configs)

class TrainState(train_state.TrainState):
    loss_fn: Callable = flax.struct.field(pytree_node=False)
    get_feat_extract_output_lengths: Callable = flax.struct.field(pytree_node=False)
    lr_scheduler: Callable = flax.struct.field(pynode_tree=False)


def training_step(
    state: train_state.TrainState,
    dropout_rng: jnp.DeviceArray,
    batch: Dict[str, jnp.DeviceArray],
) -> TrainingStepOutput:
    new_drp_rng, drp_rng = jax.random.split(dropout_rng, num=2)

    def loss_fn(params):
        labels = batch.pop("labels")
        label_paddings = batch.pop("label_paddings")

        outputs = state.apply_fn(
            **batch,
            params=params,
            dropout_rng=drp_rng,
            train=True,
            freeze_feature_encoder=True
        )
        seqlen = outputs.logits.shape[1]

        input_lengths = jnp.sum(batch["attention_mask"], axis=1)
        input_lengths = state.get_feat_extract_output_lengths(input_lengths)
        logit_paddings = input_lengths[..., None] <= jnp.arange(seqlen)

        # taking mean is fine as long as batches are equally distributed
        return state.loss_fn(
            outputs.logits, logit_paddings, labels, label_paddings
        ).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    return TrainingStepOutput(
        state=new_state,
        dropout_rng=new_drp_rng,
        loss=jax.lax.pmean(loss, axis_name="batch"),
        lr=state.lr_scheduler(state.step),
    )


def validation_step(
    state: train_state.TrainState, batch: Dict[str, jnp.DeviceArray]
) -> ValidationStepOutput:

    labels = batch.pop("labels")
    label_paddings = batch.pop("label_paddings")
    batch.pop("mask_time_indices", None)

    input_lengths = jnp.sum(batch["attention_mask"], axis=1)
    input_lengths = state.get_feat_extract_output_lengths(input_lengths)

    outputs = state.apply_fn(**batch, params=state.params, train=False)

    seqlen = outputs.logits.shape[1]
    logit_paddings = input_lengths[..., None] <= jnp.arange(seqlen)

    loss = state.loss_fn(outputs.logits, logit_paddings, labels, label_paddings).mean()
    loss = jax.lax.pmean(loss, axis_name="batch")

    return ValidationStepOutput(loss=loss)


class SpecAugmentConfig(BaseModel):
    shape: Tuple[int, int]
    mask_time_prob: float
    mask_time_span: int
    min_masks: int


@dataclasses.dataclass
class DataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    audio_maxlen: Optional[int] = None
    text_maxlen: Optional[int] = None
    spec_augment_config: Optional[SpecAugmentConfig] = None
    get_feat_extract_output_lengths: Callable = None

    def __call__(self, batch: List[Dict[str, Any]]):
        audio = [sample["audio"]["array"] for sample in batch]
        text = [sample["text"] for sample in batch]

        # TODO: explore other padding options in JAX (special dynamic padding?)
        audio = self.feature_extractor(
            audio,
            padding="max_length",
            max_length=self.audio_maxlen,
            truncation=True,
            return_tensors="np",
            sampling_rate=16000,
        )
        targets = self.tokenizer(
            text,
            max_length=self.text_maxlen,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        labels = targets["input_ids"]
        label_paddings = (targets["attention_mask"] == 0).astype(np.int32)

        outputs = {
            "input_values": audio["input_values"],
            "attention_mask": audio["attention_mask"],
            "labels": labels,
            "label_paddings": label_paddings,
        }

        if self.spec_augment_config is not None:
            input_lengths = np.sum(audio["attention_mask"], axis=1)
            assert self.get_feat_extract_output_lengths is not None
            input_lengths = self.get_feat_extract_output_lengths(input_lengths)
            seqlen = self.get_feat_extract_output_lengths(self.audio_maxlen)
            attention_mask = input_lengths[:, None] > np.arange(seqlen)

            outputs["mask_time_indices"] = _compute_mask_indices(
                self.spec_augment_config.shape,
                self.spec_augment_config.mask_time_prob,
                self.spec_augment_config.mask_time_span,
                attention_mask=attention_mask,
                min_masks=self.spec_augment_config.min_masks,
            )

        return outputs


model_id = configs["model"]["pretrained_id"]
model = FlaxWav2Vec2ForCTC.from_pretrained(model_id)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)

trainer_config = TrainerConfig.from_dict(configs["trainer"])
batch_size = trainer_config.batch_size_per_device * jax.device_count()

audio_maxlen = configs["datacollator"]["audio_maxlen"]
text_maxlen = configs["datacollator"]["text_maxlen"]
spec_augment_config = SpecAugmentConfig(
    shape=(batch_size, model._get_feat_extract_output_lengths(audio_maxlen)),
    mask_time_prob=configs["datacollator"]["mask_time_prob"],
    mask_time_span=configs["datacollator"]["mask_time_span"],
    min_masks=configs["datacollator"]["min_masks"],
)

collate_fn = DataCollator(
    feature_extractor,
    tokenizer,
    audio_maxlen=audio_maxlen,
    text_maxlen=text_maxlen,
    spec_augment_config=spec_augment_config,
    get_feat_extract_output_lengths=model._get_feat_extract_output_lengths,
)

save_fn = partial(
    hf_save_fn,
    model_save_fn=model.save_pretrained,
    feature_extractor_save_fn=feature_extractor.save_pretrained,
    tokenizer_save_fn=tokenizer.save_pretrained,
    push_to_hub=False,
)

trainer = training.Trainer(
    config=trainer_config,
    training_step=training_step,
    validation_step=validation_step,
    train_pmap_kwargs={"axis_name": "batch", "donate_argnums": (0, 1)},
    val_pmap_kwargs={"axis_name": "batch"},
    collate_fn=collate_fn,
    model_save_fn=save_fn,
)


train_data = interleave_datasets(
    [
        load_dataset(
            configs["data"]["name"],
            split.split(".", 1)[0],
            split=split.split(".", 1)[1],
            streaming=configs["data"]["streaming"],
        )
        for split in configs["data"]["train_splits"]
    ]
)
val_data = interleave_datasets(
    [
        load_dataset(
            configs["data"]["name"],
            split,
            split="validation",
            streaming=configs["data"]["streaming"],
        )
        for split in configs["data"]["validation"]
    ]
)

lr_scheduler = linear_scheduler_with_warmup(
    configs["optax"]["lr"],
    configs["optax"]["init_lr"],
    configs["optax"]["warmup_steps"],
    configs["optax"]["num_steps"],
)
tx = create_tx(lr_scheduler, configs["optax"]["weight_decay"])

state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=tx,
    loss_fn=partial(optax.ctc_loss, blank_id=tokenizer.pad_token_id),
    get_feat_extract_output_lengths=model._get_feat_extract_output_lengths,
    lr_scheduler=lr_scheduler,
)

try:
    new_state = trainer.train(state, train_data, val_data, wandb_configs=configs)
except KeyboardInterrupt:
    print("Interrupting training through KEYBOARD!!")
