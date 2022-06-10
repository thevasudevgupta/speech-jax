from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import wandb
import yaml
from datasets import IterableDataset
from flax import jax_utils, struct
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm.auto import tqdm

from .data_utils import HFIterableDataLoader

# from huggingface_hub import Repository


PathType = Union[Path, str]
OPTIMIZER_STATE_PATH = "optim_state.msgpack"
MODEL_PATH = "flax_model.msgpack"
TRAINING_STATE_PATH = "training_state.yaml"


@struct.dataclass
class TrainingStepOutput:
    state: train_state.TrainState
    dropout_rng: jnp.DeviceArray
    loss: jnp.DeviceArray

@struct.dataclass
class ValidationStepOutput:
    loss: jnp.DeviceArray


class TrainerConfig(BaseModel):
    max_epochs: int
    train_batch_size_per_device: int
    eval_batch_size_per_device: int
    wandb_project_name: str
    epochs_save_dir: Optional[str] = None
    logging_steps: int


class Trainer(BaseModel):
    config: TrainerConfig
    training_step: Callable
    validation_step: Callable
    pmap_kwargs: Dict[str, Any] = {}
    collate_fn: Optional[Callable] = None
    lr_scheduler: Callable = None

    # input signature has `save_dir` & `params`
    model_save_fn: Optional[Callable] = None

    def train(
        self,
        state: train_state.TrainState,
        train_data: IterableDataset,
        val_data: IterableDataset,
        seed: int = 0,
    ):
        logger = wandb.init(project=self.config.wandb_project_name, config=self.config.dict())
        # jax.start_trace("./tensorboard")

        train_batch_size = self.config.train_batch_size_per_device * jax.device_count()
        eval_batch_size = self.config.eval_batch_size_per_device * jax.device_count()

        train_data = HFIterableDataLoader(
            train_data, batch_size=train_batch_size, collate_fn=self.collate_fn
        )
        train_data.shuffle(seed)

        val_data = HFIterableDataLoader(
            val_data, batch_size=eval_batch_size, collate_fn=self.collate_fn
        )

        state = jax_utils.replicate(state)
        training_step = jax.pmap(self.training_step, **self.pmap_kwargs)
        validation_step = jax.pmap(self.validation_step, **self.pmap_kwargs)

        rng = jax.random.PRNGKey(seed)
        dropout_rng = jax.random.split(rng, jax.device_count())

        for epoch in range(self.config.max_epochs):
            tr_loss, avg_tr_loss = jnp.array(0), jnp.array(0)
            train_data.set_epoch(epoch)

            pbar = tqdm(enumerate(train_data), desc=f"Running epoch-{epoch}")
            for step, batch in pbar:
                batch = shard(batch)

                # TODO: logging old step lr
                lr = self.lr_scheduler(jax_utils.unreplicate(state.step))
                
                outputs = training_step(state, dropout_rng, batch)
                state, dropout_rng = outputs.state, outputs.dropout_rng
                loss = jax_utils.unreplicate(outputs.loss)

                tr_loss += loss
                avg_tr_loss += loss

                if (step + 1) % self.config.logging_steps == 0:
                    logs = {
                            "tr_loss": tr_loss.item() / self.config.logging_steps,
                            "avg_tr_loss": avg_tr_loss.item() / (step + 1),
                            "lr": lr.item(),
                        }
                    pbar.set_postfix(**logs)
                    logger.log(logs)
                    tr_loss = jnp.array(0)

            if self.config.epochs_save_dir is not None:
                self.save_checkpoint(
                    jax_utils.unreplicate(state), Path(self.config.epochs_save_dir, f"epoch-{epoch}")
                )

            val_steps, val_loss = 0, jnp.array(0)
            for batch in tqdm(val_data):
                batch = shard(batch)
                outputs = validation_step(state, batch)
                val_loss += jax_utils.unreplicate(outputs.loss)
                val_steps += 1
            logger.log({"val_loss": val_loss.item() / val_steps, "epoch": epoch})

        # jax.stop_trace()

        return jax_utils.unreplicate(state)

    def save_checkpoint(
        self,
        state: train_state.TrainState,
        ckpt_dir: PathType,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        # state must be unreplicated before passing it

        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        # repo = Repository(ckpt_dir, clone_from=self.config.clone_from, use_auth_token=True)
        # with repo.commit("speech_jax 🔥"):

        # training_state = {
        #     "config": self.config.dict(),
        #     "extra": extra,
        # }
        # with open(ckpt_dir / TRAINING_STATE_PATH, "w") as f:
        #     yaml.dump(training_state, f)

        if self.model_save_fn is not None:
            self.model_save_fn(ckpt_dir, state.params)
        else:
            with open(ckpt_dir / MODEL_PATH, "wb") as f:
                f.write(to_bytes(state.params))
        with open(ckpt_dir / OPTIMIZER_STATE_PATH, "wb") as f:
            f.write(to_bytes(state.opt_state))

        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: PathType):
        ...
