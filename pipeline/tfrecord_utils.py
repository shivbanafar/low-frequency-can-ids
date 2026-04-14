"""TFRecord serialization compatible with utils.read_tfrecord (TensorFlow 1 data pipeline).

Uses the lightweight `tfrecord` package (no TensorFlow import) so preprocessing does not
block on loading the full TF runtime.
"""
import numpy as np
from tqdm import tqdm
from tfrecord.writer import TFRecordWriter


def serialize_example(x, y):
    """Return bytes of a tf.train.Example protobuf (same keys/shapes as before)."""
    x = np.asarray(x, dtype=np.int64).flatten().tolist()
    y = int(np.asarray(y).flat[0])
    return TFRecordWriter.serialize_tf_example(
        {
            "input_features": (x, "int"),
            "label": ([y], "int"),
        }
    )


def write_tfrecord(data, filename):
    writer = TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        x = np.asarray(row["features"], dtype=np.int64).flatten().tolist()
        y = int(row["label"])
        writer.write(
            {
                "input_features": (x, "int"),
                "label": ([y], "int"),
            }
        )
    writer.close()
