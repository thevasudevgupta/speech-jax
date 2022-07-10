import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tqdm.auto import tqdm

SPEECH_DTYPE = tf.float32
LABEL_DTYPE = tf.string

CLEAN_SPLITS = ["train.100", "train.360", "validation", "test"]
OTHER_SPLITS = ["train.500", "validation", "test"]
# python3 src/speech_jax/make_tfrecords.py -c clean -s validation


def create_tfrecord(speech_tensor: tf.Tensor, label_tensor: tf.Tensor):
    speech_tensor = tf.cast(speech_tensor, SPEECH_DTYPE)
    label_tensor = tf.cast(label_tensor, LABEL_DTYPE)

    speech_bytes = tf.io.serialize_tensor(speech_tensor).numpy()
    label_bytes = tf.io.serialize_tensor(label_tensor).numpy()

    feature = {
        "speech": tf.train.Feature(bytes_list=tf.train.BytesList(value=[speech_bytes])),
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def read_tfrecord(record: tf.train.Example):
    desc = {
        "speech": tf.io.FixedLenFeature((), tf.string),
        "label": tf.io.FixedLenFeature((), tf.string),
    }
    record = tf.io.parse_single_example(record, desc)

    speech = tf.io.parse_tensor(record["speech"], out_type=SPEECH_DTYPE)
    label = tf.io.parse_tensor(record["label"], out_type=LABEL_DTYPE)

    return speech, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "CLI to convert LibriSpeech dataset (from `huggingface-datasets`) to .tfrecords format"
    )
    parser.add_argument("-c", "--split_config", required=True, type=str)
    parser.add_argument("-s", "--split_name", required=True, type=str)
    parser.add_argument("-d", "--tfrecords_dir", default=None, type=str)
    parser.add_argument("-n", "--num_shards", default=1, type=int)

    args = parser.parse_args()

    tfrecords_dir = Path(
        f"{args.split_config}.{args.split_name}"
        if args.tfrecords_dir is None
        else args.tfrecord_dir
    )
    tfrecords_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset("librispeech_asr", args.split_config, split=args.split_name)
    print(dataset)

    # shards the TFrecords into several files (since overall dataset size is approx 280 GB)
    # this will help TFRecordDataset to read shards in parallel from several files
    # Docs suggest to keep each shard around 100 MB in size, so choose `num_shards` accordingly
    num_records_per_file = len(dataset) // args.num_shards
    file_names = [
        str(tfrecords_dir / f"{tfrecords_dir}-{i}.tfrecord")
        for i in range(args.num_shards)
    ]

    writers = [tf.io.TFRecordWriter(file_name) for file_name in file_names]

    # following loops runs in O(n) time (assuming n = num_samples & for every tfrecord prepartion_take = O(1))
    i, speech_stats, label_stats = 0, [], []
    pbar = tqdm(dataset, total=len(dataset), desc=f"Preparing {file_names[i]} ... ")
    for j, inputs in enumerate(pbar):
        speech, label = inputs["audio"]["array"], inputs["text"]

        speech_stats.append(len(speech))
        label_stats.append(len(label))

        speech, label = tf.convert_to_tensor(speech), tf.convert_to_tensor(label)
        tf_record = create_tfrecord(speech, label)

        writers[i].write(tf_record)
        if (j + 1) % num_records_per_file == 0:
            if i == len(file_names) - 1:
                # last file will have extra samples
                continue
            writers[i].close()
            i += 1
            pbar.set_description(f"Preparing {file_names[i]} ... ")
    writers[-1].close()

    print(f"Total {len(dataset)} tfrecords are sharded in `{tfrecords_dir}`")
    print("############# Data Stats #############")
    print(
        {
            "speech_min": min(speech_stats),
            "speech_mean": sum(speech_stats) / len(speech_stats),
            "speech_max": max(speech_stats),
            "speech_std": np.std(speech_stats),
            "label_min": min(label_stats),
            "label_mean": sum(label_stats) / len(label_stats),
            "label_max": max(label_stats),
            "label_std": np.std(label_stats),
        }
    )
    print("######################################")
