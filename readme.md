Something exciting WIP

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

config = training.TrainerConfig(...)
trainer = training.Trainer(config, ...)
state = trainer.train(...)
# state will be lost if you don't return it (hence, make sure you return it)

trainer.save_checkpoint(state, "ckpt_dir")
```
