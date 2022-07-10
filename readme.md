## Trained Models

| Model ID | WER (test-clean) |
|----------|------------------|
| [speech_jax_wav2vec2-large-lv60_960h](https://huggingface.co/vasudevgupta/speech_jax_wav2vec2-large-lv60_960h) | 3.38% |
| [speech_jax_wav2vec2-large-lv60_100h](https://huggingface.co/vasudevgupta/speech_jax_wav2vec2-large-lv60_100h) | 5.5% |

## Running Experiments

**Installation**

```bash
git clone https://github.com/vasudevgupta7/speech-jax.git
pip3 install -e .

# JAX & tensorflow should be installed by user depending on your hardware
# https://github.com/google/jax#pip-installation-google-cloud-tpu
# you don't need to install JAX & tensorflow if you are running training on Cloud TPUs
```

**Converting librispeech data to tfrecords**

```python
# there are many librispeech splits available 
# you can set `-c` & `-s` flags appropriately to download and convert those splits into tfrecords
python3 src/speech_jax/make_tfrecords.py -c clean -s train.100 -n 100
```

**Uploading tfrecords to GCS bucket**

```python
gsutil -m cp -r clean.train.100 gs://librispeech_jax/

# similarly, you can copy other directories to your GCS bucket
```

**Launching Cloud TPUs**

```python
# setting env variables for later use
export TPU_NAME=jax-models
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v3-8
export RUNTIME_VERSION=v2-alpha

# create TPU VM
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION}

# ssh TPU VM
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE}
```

**Starting training**

```bash
# switch to relevant directory
cd projects

# following command will finetune Wav2Vec2-large model on librispeech-960h dataset
python3 finetune_wav2vec2.py configs/wav2vec2_asr.yaml

# final model is saved in the huggingface format 
# => you can load it directly using `FlaxAutoModel.from_pretrained`
```

## Thank You

* ML Google Developers Experts program and TPU Cloud Research for providing free TPU resources for experimentation
