***This project is work in progress and I don't recommend you to use this project now. Something stable would be released soon and then you would be able to use this project for your tasks.***

## Models

| Model ID | WER (test-clean) |
|----------|------------------|
| [speech_jax_wav2vec2-large-lv60_100h](https://huggingface.co/vasudevgupta/speech_jax_wav2vec2-large-lv60_100h) | 5.5% |
| [speech_jax_wav2vec2-large-lv60_960h](https://huggingface.co/vasudevgupta/speech_jax_wav2vec2-large-lv60_960h) |  |

## Running experiments

```bash
# following command will finetune Wav2Vec2-large model on Librispeech-960h dataset
python3 projects/finetune_wav2vec2.py

# following command will pre-train Wav2Vec2-base model on Librispeech-960h dataset
python3 projects/pretrain_wav2vec2.py

# final model is saved in the huggingface format 
# => you can load it directly using `FlaxAutoModel.from_pretrained`
```

### For development locally

```bash
git clone https://github.com/vasudevgupta7/speech-jax.git
pip3 install -e ".[common]"

# JAX should be installed by user depending on your hardware
# https://github.com/google/jax#pip-installation-google-cloud-tpu
```

### Running tests

```bash
pytest -sv tests/
```

### Usage

```python
from speech_jax import training
from flax.tx_utils import create_tx, linear_scheduler_with_warmup
from flax.training import train_state

config = training.TrainerConfig(...)
trainer = training.Trainer(config, ...)

lr = linear_scheduler_with_warmup(...)
tx = create_tx(lr, ...)

state = train_state.TrainState.create(..., tx=tx)
state = trainer.train(state, train_data, val_data)
# state will be lost if you don't return it (hence, make sure you return it)

trainer.save_checkpoint(state, "ckpt_dir")
```

## Thank You

* TPU Cloud Research for providing free TPU resources for experimentation
