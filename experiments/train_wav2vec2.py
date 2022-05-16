import jax
import flax

import jax.numpy as jnp
from flax import traverse_util, shard
from flax.training import train_state
import optax
from typing import Callable

from transformers import FlaxWav2Vec2ForCTC

model_id = "facebook/wav2vec2-base"
model = FlaxWav2Vec2ForCTC.from_pretrained(model_id)


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    loss_fn: Callable = flax.struct.field(pytree_node=False)


def create_tx(lr, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {k: (v[-1] != "bias" and v[-2:] != ("LayerNorm", "scale")) for k, v in params.items()}
        return traverse_util.unflatten_dict(mask)
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask)
    return tx


state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=create_tx(1e-4, 1e-4),
    loss_fn=optax.ctc_loss,
)

import dataclasses
from tqdm.auto import tqdm
from flax import jax_utils

from functools import partial


@partial(jax.pmap, axis_name="batch")
def training_step(batch, state, drp_rng: jnp.DeviceArray):
    new_drp_rng, drp_rng = jax.random.split(drp_rng, num=2)

    def loss_fn(params):
        targets = batch.pop("targets")
        outputs = state.apply({"params": params}, **batch, dropout_rng=drp_rng, train=True)
        return state.loss_fn(targets, outputs)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    loss = state.apply_gradient(grads)

    return loss, new_drp_rng


@partial(jax.pmap, axis_name="batch")
def validation_step(batch, state):
    targets = batch.pop("targets")
    outputs = state.apply({"params": state.params}, **batch, train=False)
    loss = state.loss_fn(targets, outputs)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return loss


@dataclasses.dataclass
class TrainerConfig:
    max_epochs: int
    wandb_project_name: str = "speech-JAX"

import wandb

@dataclasses.dataclass
class Trainer:
    config: TrainerConfig
    datacollator: Callable
    training_step: Callable
    validation_step: Callable
    state: train_state.TrainState

    def train(self, train_data, val_data):
        logger = wandb.init(project=self.config.wandb_project_name)

        state = jax_utils.replicate(state)

        rng = jax.random.PRNGKey(0)
        drp_rng = jax.random.split(rng, jax.device_count())

        for epoch in range(self.config.max_epochs):
            tr_loss = jnp.array(0)
            for step, batch in tqdm(enumerate(train_data)):
                batch = self.datacollator(batch)
                batch = shard(batch)
                loss, drp_rng = self.training_step(batch, state, drp_rng)

                tr_loss += jax_utils.unreplicate(loss)
                if step % self.config.logging_steps == 0:
                    logger.log({"tr_loss": tr_loss.item()})
                    tr_loss = jnp.array(0)

            val_loss = self.evaluate(val_data)
            logger.log({"val_loss": val_loss.item(), "epoch": epoch})

    def evaluate(self, data):
        val_loss = jnp.array(0)
        for batch in tqdm(data):
            batch = self.datacollator(batch)
            batch = shard(batch)
            loss = self.validation_step(batch, state)
            val_loss += jax_utils.unreplicate(loss)
        return val_loss


def collate_fn(batch):
    return batch


trainer_config = TrainerConfig(
    max_epochs=30,
    wandb_project_name="speech-JAX",
)


trainer = Trainer(
    config=trainer_config,
    datacollator=collate_fn,
    training_step=training_step,
    validation_step=validation_step,
    state=state,
)
