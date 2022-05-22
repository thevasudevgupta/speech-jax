import dataclasses
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.training import train_state
from transformers import (FlaxWav2Vec2ForCTC, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor)

from speech_jax import training


class TrainState(train_state.TrainState):
    loss_fn: Callable = flax.struct.field(pytree_node=False)
    get_feat_extract_output_lengths: Callable = flax.struct.field(pytree_node=False)


def create_tx(lr, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {
            k: (v[-1] != "bias" and v[-2:] != ("LayerNorm", "scale"))
            for k, v in params.items()
        }
        return traverse_util.unflatten_dict(mask)

    tx = optax.adamw(
        learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask
    )
    return tx


def training_step(
    state: train_state.TrainState, drp_rng: jnp.DeviceArray, batch: Dict[str, jnp.Array]
):
    new_drp_rng, drp_rng = jax.random.split(drp_rng, num=2)

    def loss_fn(params):
        labels = batch.pop("labels")
        label_paddings = batch.pop("label_paddings")

        input_lengths = jnp.sum(batch["attention_mask"], axis=1)
        input_lengths = state.get_feat_extract_output_lengths(input_lengths)

        seqlen = batch["attention_mask"].shape[-1]
        logit_paddings = input_lengths[..., None] <= jnp.arange(seqlen)

        logits = state.apply(
            {"params": params}, **batch, dropout_rng=drp_rng, train=True
        )

        return state.loss_fn(logits, logit_paddings, labels, label_paddings)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradient(grads)

    return new_state, new_drp_rng, loss


def validation_step(state: train_state.TrainState, batch: Dict[str, jnp.Array]):
    labels = batch.pop("labels")
    label_paddings = batch.pop("label_paddings")

    input_lengths = jnp.sum(batch["attention_mask"], axis=1)
    input_lengths = state.get_feat_extract_output_lengths(input_lengths)

    seqlen = batch["attention_mask"].shape[-1]
    logit_paddings = input_lengths[..., None] <= jnp.arange(seqlen)

    logits = state.apply({"params": state.params}, **batch, train=False)

    loss = state.loss_fn(logits, logit_paddings, labels, label_paddings)

    loss = jax.lax.pmean(loss, axis_name="batch")
    return state, loss


@dataclasses.dataclass
class DataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    audio_max_len: Optional[int] = None
    text_max_len: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]):
        audio = [sample["audio"]["array"] for sample in batch]
        text = [sample["text"] for sample in batch]

        # TODO: explore other padding options in JAX (special dynamic padding?)
        audio = self.feature_extractor(
            audio,
            padding="max_length",
            max_length=self.audio_max_len,
            truncation=True,
            return_tensors="np",
        )
        targets = self.tokenizer(
            text,
            max_length=self.text_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        labels = targets["input_ids"]
        label_paddings = np.cast(targets["attention_mask"] == 0, np.int32)

        return {
            "input_values": audio["input_values"],
            "attention_mask": audio["attention_mask"],
            "labels": labels,
            "label_paddings": label_paddings,
        }


class TrainerConfig(training.TrainerConfig):
    lr: float
    weight_decay: float


# TODO (for fine-tuning):
# need to freeze feature extractor

model_id = "facebook/wav2vec2-large-lv60"
model = FlaxWav2Vec2ForCTC.from_pretrained(model_id)

trainer_config = TrainerConfig(
    max_epochs=2,
    train_batch_size_per_device=2,
    eval_batch_size_per_device=2,
    wandb_project_name="speech-JAX",
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
collate_fn = DataCollator(
    feature_extractor, tokenizer, audio_max_len=256000, text_max_len=16
)

trainer = training.Trainer(
    config=trainer_config,
    training_step=training_step,
    validation_step=validation_step,
    pmap_kwargs={"axis_name": "batch", "donate_argnums": (0, 1)},
    collate_fn=collate_fn,
)


from datasets import interleave_datasets, load_dataset

train_data = [
    load_dataset("librispeech_asr", "clean", split="train.100", streaming=True),
    # load_dataset("librispeech_asr", "clean", split="train.360", streaming=True),
    # load_dataset("librispeech_asr", "other", split="train.500", streaming=True),
]
train_data = interleave_datasets(train_data)
val_data = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)


state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=create_tx(trainer_config.lr, trainer_config.weight_decay),
    loss_fn=partial(optax.ctc_loss, blank_id=tokenizer.pad_token_id),
    get_feat_extract_output_lengths=model._get_feat_extract_output_lengths,
)

try:
    state = trainer.train(state, train_data, val_data)
except KeyboardInterrupt:
    print("Interrupting training through KEYBOARD!!")

model.save_pretrained("final-model", params=state.params)
