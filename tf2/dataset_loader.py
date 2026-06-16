import tensorflow as tf
from pathlib import Path
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, image_size):

    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0

    return img


def create_dataset(data_dir, image_size, csv_path):

    df = pd.read_csv(csv_path)
    

    image_paths = []
    labels = []

    # CSV = Single Source of Truth
    for _, row in df.iterrows():

        # row["image_right"] holt den wert aus der aktuellen csv-Zeile, Path(data_dir) erzeugt ein Pathobjekt, / -> joint beides zu neuem Path
        image_paths.append(
            str(Path(data_dir) / row["image_right"])
        )

        labels.append([
            row["l_shoulder_z"],
            row["l_shoulder_y"],
            row["l_arm_x"],
            row["l_elbow_y"],
            row["l_wrist_z"],
            row["l_wrist_x"]
        ])

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def process(path, label):
        image = load_image(path, image_size)
        label = tf.cast(label, tf.float32)
        return image, label

    ds = ds.map(
        process,
        num_parallel_calls=AUTOTUNE
    )

    """image_paths = list(Path(data_dir).glob("*.png"))
    image_paths = [str(p) for p in image_paths]
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(
        lambda x: load_image(x, image_size),
        num_parallel_calls=AUTOTUNE
    )"""

    return ds

def get_dataset_size(csv_path):
    df = pd.read_csv(csv_path)
    return len(df)