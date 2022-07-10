from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple, Union

import tensorflow as tf

from speech_jax.make_tfrecords import LABEL_DTYPE, SPEECH_DTYPE, read_tfrecord
from speech_jax.training import BaseConfig

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class BaseDataloader:
    batch_size: int
    audio_maxlen: int
    labels_maxlen: int

    audio_pad_id: float = 0.0
    labels_pad_id: int = 0
    buffer_size: int = 10000

    def batchify(
        self, dataset: tf.data.Dataset, seed: int = None, drop_remainder: bool = True
    ):
        # shuffling for training
        if seed is not None:
            dataset.shuffle(self.buffer_size, seed=seed)

        padded_shapes = (self.audio_maxlen, self.labels_maxlen)
        padding_values = (self.audio_pad_id, self.labels_pad_id)

        dataset = dataset.map(self.restrict_to_maxlen)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=drop_remainder,
        )

        return dataset.prefetch(AUTOTUNE)

    # def restrict_to_maxlen(self, speech: tf.Tensor, labels: tf.Tensor):
    #     """This must be called before doing padding"""
    #     speech, labels = speech[: self.audio_maxlen], labels[: self.labels_maxlen]
    #     return speech, labels


class LibriSpeechDataLoaderConfig(BaseConfig):
    tfrecords: List[str] = field(default_factory=lambda: ["clean.test/*.tfrecord"])

    batch_size: int = 16
    buffer_size: int = 10000

    audio_maxlen: int = 256000
    audio_pad_id: float = 0.0

    labels_maxlen: int = 256
    labels_pad_id: int = 0


class LibriSpeechDataLoader(BaseDataloader):
    def __init__(
        self, config: LibriSpeechDataLoaderConfig, required_sample_rate: int = 16000
    ):
        super().__init__(
            batch_size=config.batch_size,
            audio_maxlen=config.args.audio_maxlen,
            labels_maxlen=config.args.labels_maxlen,
            audio_pad_id=config.args.audio_pad_id,
            labels_pad_id=config.args.labels_pad_id,
            buffer_size=config.args.buffer_size,
        )

        self.tfrecords = args.tfrecords
        self.required_sample_rate = required_sample_rate

    def __call__(
        self, seed: int = None, drop_remainder: bool = True
    ) -> tf.data.Dataset:

        print(f"Reading tfrecords from {self.tfrecords}", end=" ... ")
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
        print("Done!")

        return self.batchify(dataset, seed=seed, drop_remainder=drop_remainder)


if __name__ == "__main__":
    config = LibriSpeechDataLoaderConfig(batch_size=2)
    dataloader = LibriSpeechDataLoader(config)
    for sample in dataloader:
        print(sample)
        break
