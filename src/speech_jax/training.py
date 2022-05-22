import dataclasses
from pathlib import Path
from typing import Callable, Union, Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import wandb
from datasets import IterableDataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm.auto import tqdm

from flax.serialization import to_bytes, from_bytes
import yaml

PathType = Union[Path, str]
OPTIMIZER_STATE_PATH = "optim_state.msgpack"
MODEL_PATH = "flax_model.msgpack"
TRAINING_STATE_PATH = "training_state.yaml"


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
    epochs_save_dir: str = "epochs"


@dataclasses.dataclass
class Trainer:
    config: TrainerConfig
    training_step: Callable
    validation_step: Callable
    pmap_kwargs: Dict[str, Any] = {}
    collate_fn: Optional[Callable] = None

    def train(
        self,
        state: train_state.TrainState,
        train_data: IterableDataset,
        val_data: IterableDataset,
        seed: int = 0,
    ):
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
        training_step = jax.pmap(self.training_step, **self.pmap_kwargs)
        validation_step = jax.pmap(self.validation_step, **self.pmap_kwargs)

        rng = jax.random.PRNGKey(seed)
        drp_rng = jax.random.split(rng, jax.device_count())

        epochs_save_dir = Path(self.config.epochs_save_dir)
        epochs_save_dir.mkdir(exist_ok=True)

        for epoch in range(self.config.max_epochs):
            tr_loss, avg_tr_loss = jnp.array(0), jnp.array(0)
            for step, batch in tqdm(enumerate(train_data)):
                batch = shard(batch)
                state, drp_rng, loss = training_step(state, drp_rng, batch)
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

            val_loss = jnp.array(0)
            for batch in tqdm(val_data):
                batch = shard(batch)
                state, loss = validation_step(state, batch)
                val_loss += jax_utils.unreplicate(loss)
            logger.log({"val_loss": val_loss.item(), "epoch": epoch})

            self.save_checkpoint(jax_utils.unreplicate(state), epochs_save_dir / f"epoch-{epoch}")

        return jax_utils.unreplicate(state)

    def save_checkpoint(self, state: train_state.TrainState, ckpt_dir: PathType, extra: Dict[str, Any]) -> Path:
        # state must be unreplicated

        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True)

        training_state = {
            "config": self.config.to_dict(),
            "extra": extra,
        }
        yaml.dump(training_state, TRAINING_STATE_PATH)

        with open(ckpt_dir / MODEL_PATH) as f:
            f.write(to_bytes(state.params))
        with open(ckpt_dir / OPTIMIZER_STATE_PATH):
            f.write(to_bytes(state.opt_state))

        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: PathType):
        ...
