"""Project-wide constants and settings."""
from pathlib import Path

BASE_DIR = Path(__file__).parent

train_mount_point = Path('/') / "train"
validation_mount_point = Path('/') / "validation"
test_mount_point = Path('/') / "test"

data_paths = {
        "classes": {
          "descriptions": BASE_DIR / "data" / "oidv6-class-descriptions.csv",
          "trainable": BASE_DIR / "data" / "oidv6-classes-trainable.txt",
        },
        "train": {
            "image_locations": BASE_DIR / "data" / "oidv6-train-images-with-labels-with-rotation.csv",
            "human_labels": BASE_DIR / "data" / "oidv6-train-annotations-human-imagelabels.csv",
            "machine_labels": BASE_DIR / "data" / "train-annotations-machine-imagelabels.csv",
        },
        "validation": {
            "image_locations": BASE_DIR / "data" / "validation-images-with-rotation.csv",
            "human_labels": BASE_DIR / "data" / "validation-annotations-human-imagelabels.csv",
            "machine_labels": BASE_DIR / "data" / "validation-annotations-machine-imagelabels.csv",
        },
        "test": {
            "image_locations": BASE_DIR / "data" / "test-images-with-rotation.csv",
            "human_labels": BASE_DIR / "data" / "test-annotations-human-imagelabels.csv",
            "machine_labels": BASE_DIR / "data" / "test-annotations-machine-imagelabels.csv",
        }
    }
