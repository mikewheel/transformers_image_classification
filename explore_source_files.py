"""Run this with the debugger on in your IDE of choice to inspect the contents of each of these in disk."""

from pathlib import Path

import pandas

if __name__ == "__main__":
    
    split = "train"  # or validation or test
    
    BASE_DIR = Path(__file__).parent

    data_paths = {
        "train": {
            "image_locations": BASE_DIR / "data" / "oidv6-train-images-with-labels-with-rotation.csv",
            "human_labels": BASE_DIR / "data" / "oidv6-train-annotations-human-imagelabels.csv",
            "machine_labels": BASE_DIR / "data" / "train-annotations-machine-imagelabels.csv"
        },
        "validation": {
            "image_locations": BASE_DIR / "data" / "validation-images-with-rotation.csv",
            "human_labels": BASE_DIR / "data" / "validation-annotations-human-imagelabels.csv",
            "machine_labels": BASE_DIR / "data" / "validation-annotations-machine-imagelabels.csv"
        },
        "test": {
            "image_locations": BASE_DIR / "data" / "test-images-with-rotation.csv",
            "human_labels": BASE_DIR / "data" / "test-annotations-human-imagelabels.csv",
            "machine_labels": BASE_DIR / "data" / "test-annotations-machine-imagelabels.csv"
        }
    }
    
    class_descriptions = pandas.read_csv("data/oidv6-class-descriptions.csv")
    trainable_classes = pandas.read_csv("data/oidv6-classes-trainable.txt")
    
    # These ones are really big and should handled with Dask in full usage
    image_locations = pandas.read_csv(data_paths[split]["image_locations"], nrows=100)
    human_training_labels = pandas.read_csv(data_paths[split]["human_labels"], nrows=100)
    machine_training_labels = pandas.read_csv(data_paths[split]["machine_labels"], nrows=100)
    print(class_descriptions.columns)
