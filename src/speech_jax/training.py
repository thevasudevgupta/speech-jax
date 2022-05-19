import dataclasses
from pathlib import Path
from typing import Callable, Union, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import wandb
from datasets import IterableDataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm.auto import tqdm

from flax.serialization import to_bytes, from_bytes

PathType = Union[Path, str]
OPTIMIZER_STATE_PATH = "optim_state.msgpack"
MODEL_PATH = "flax_model.msgpack"

class DataLoader:
    def __init__(self, dataset: IterableDataset, batch_size: int = 1, collate_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self) -> Union[Tuple[jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]:
        batch = []
        for i, sample in enumerate(self.dataset):
            batch.append(sample)

            if (i + 1) % self.batch_size == 0:
                if self.collate_fn is not None:
                    batch = self.collate_fn(batch)

                yield batch
                batch = []


@dataclasses.dataclass
class TrainerConfig:
    max_epochs: int
    train_batch_size_per_device: int
    eval_batch_size_per_device: int
    wandb_project_name: str = "speech-JAX"
    epochs_save_dir: str = ""


@dataclasses.dataclass
class Trainer:
    config: TrainerConfig
    collate_fn: Callable
    training_step: Callable
    validation_step: Callable

    def train(self, state: train_state.TrainState, train_data: IterableDataset, val_data: IterableDataset):
        logger = wandb.init(project=self.config.wandb_project_name)

        train_batch_size = self.config.train_batch_size_per_device * jax.device_count()
        eval_batch_size = self.config.eval_batch_size_per_device * jax.device_count()

        train_data = DataLoader(
            train_data, batch_size=train_batch_size, collate_fn=self.collate_fn
        )
        val_data = DataLoader(
            val_data, batch_size=eval_batch_size, collate_fn=self.collate_fn
        )

        state = jax_utils.replicate(state)

        rng = jax.random.PRNGKey(0)
        drp_rng = jax.random.split(rng, jax.device_count())

        save_dir = Path(self.config.epochs_save_dir)
        save_dir.mkdir(exist_ok=True)

        for epoch in range(self.config.max_epochs):
            tr_loss, avg_tr_loss = jnp.array(0), jnp.array(0)
            for step, batch in tqdm(enumerate(train_data)):
                batch = shard(batch)
                loss, drp_rng = self.training_step(batch, state, drp_rng)
                loss = jax_utils.unreplicate(loss)

                tr_loss += loss
                avg_tr_loss += loss

                if step % self.config.logging_steps == 0:
                    logger.log(
                        {
                            "tr_loss": tr_loss.item() / self.config.logging_steps,
                            "avg_tr_loss": avg_tr_loss.item() / (step + 1),
                        }
                    )
                    tr_loss = jnp.array(0)

            val_loss = self.evaluate(state, val_data)
            logger.log({"val_loss": val_loss.item(), "epoch": epoch})

            self.save_checkpoint(state, save_dir / f"epoch-{epoch}")

        return state

    def evaluate(self, state: train_state.TrainState, data: DataLoader) -> jnp.ndarray:
        val_loss = jnp.array(0)
        for batch in tqdm(data):
            batch = shard(batch)
            loss = self.validation_step(batch, state)
            val_loss += jax_utils.unreplicate(loss)
        return val_loss

    def save_checkpoint(self, state: train_state.TrainState, ckpt_dir: PathType) -> Path:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True)
        # TODO: add logic here
        # directory saving
        # model config in config.yaml
        # training config in ...

        state = jax_utils.unreplicate(state)

        ckpt_dir / OPTIMIZER_STATE_PATH

        with open(ckpt_dir / MODEL_PATH) as f:
            f.write(to_bytes(state.params))
        with open(ckpt_dir / OPTIMIZER_STATE_PATH):
            f.write(to_bytes(state.opt_state))

        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: PathType):
        ...
