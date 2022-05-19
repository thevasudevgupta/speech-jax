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
from speech_jax.training import DataLoader, Trainer, TrainerConfig
```
