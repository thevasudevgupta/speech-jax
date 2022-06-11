import optax
from flax import traverse_util


def linear_scheduler_with_warmup(lr, init_lr, warmup_steps, num_train_steps):
    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=init_lr, end_value=lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=lr, end_value=1e-7, transition_steps=decay_steps
    )
    lr = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )
    return lr


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
