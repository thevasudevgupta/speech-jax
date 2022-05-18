import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import wandb
from datasets import IterableDataset
from flax import jax_utils, shard
from flax.training import train_state
from tqdm.auto import tqdm


class DataLoader:
    def __init__(self, dataset: IterableDataset, batch_size=1, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            if i == self.batch_size:
                break
            batch.append(sample)

        if self.collate_fn is not None:
            batch = self.collate_fn(batch)
        return batch


@dataclasses.dataclass
class TrainerConfig:
    max_epochs: int
    train_batch_size_per_device: int
    eval_batch_size_per_device: int
    wandb_project_name: str = "speech-JAX"


@dataclasses.dataclass
class Trainer:
    config: TrainerConfig
    collate_fn: Callable
    training_step: Callable
    validation_step: Callable
    state: train_state.TrainState

    def train(self, train_data: IterableDataset, val_data: IterableDataset):
        logger = wandb.init(project=self.config.wandb_project_name)

        train_batch_size = self.config.train_batch_size * jax.device_count()
        eval_batch_size = self.config.eval_batch_size * jax.device_count()

        train_data = DataLoader(
            train_data, batch_size=train_batch_size, collate_fn=self.collate_fn
        )
        val_data = DataLoader(
            val_data, batch_size=eval_batch_size, collate_fn=self.collate_fn
        )

        state = jax_utils.replicate(state)

        rng = jax.random.PRNGKey(0)
        drp_rng = jax.random.split(rng, jax.device_count())

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

            val_loss = self.evaluate(val_data, state)
            logger.log({"val_loss": val_loss.item(), "epoch": epoch})

    def evaluate(self, data: DataLoader, state: train_state.TrainState):
        val_loss = jnp.array(0)
        for batch in tqdm(data):
            batch = shard(batch)
            loss = self.validation_step(batch, state)
            val_loss += jax_utils.unreplicate(loss)
        return val_loss
