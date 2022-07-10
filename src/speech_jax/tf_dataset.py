from dataclasses import dataclass
from typing import List

import tensorflow as tf

from speech_jax.make_tfrecords import read_tfrecord


@dataclass
class TFDatasetReader:
    tfrecords: List[str]
    buffer_size: int = 10000

    def __post_init__(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = dataset

    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def set_epoch(self, seed: int):
        self.dataset = self.dataset.shuffle(self.buffer_size, seed=seed)


if __name__ == "__main__":
    dataset = TFDatasetReader(tfrecords=["../clean.test/clean.test-0.tfrecord"])
    for sample in dataset:
        print(sample)
        break
