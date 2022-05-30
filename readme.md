***This project is work in progress and I don't recommend you to use this project now. Something stable would be released soon and then you would be able to use this project for your tasks.***

### For development locally

```bash
# JAX should be installed by user before running following

git clone https://github.com/vasudevgupta7/speech-jax.git
pip3 install -e .
```

### Running tests

```bash
pytest -sv tests/
```

### Usage

```python
from speech_jax import training
from flax.tx_utils import create_tx
from flax.training import train_state

config = training.TrainerConfig(...)
trainer = training.Trainer(config, ...)

tx = create_tx(...)

state = train_state.TrainState.create(..., tx=tx)
state = trainer.train(state, train_data, val_data)
# state will be lost if you don't return it (hence, make sure you return it)

trainer.save_checkpoint(state, "ckpt_dir")
```
