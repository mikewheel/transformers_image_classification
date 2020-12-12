"""Run this with the debugger on in your IDE of choice to inspect the contents of each of these in disk."""

from dask.dataframe import read_csv
import pandas
from config import data_paths

if __name__ == "__main__":
    split = "train"  # or validation or test
    
    class_descriptions = pandas.read_csv(data_paths["classes"]["descriptions"])
    trainable_classes = pandas.read_csv(data_paths["classes"]["trainable"])
    
    # These ones are really big and should handled with Dask in full usage
    image_locations_full = read_csv(data_paths[split]["image_locations"])
    human_training_labels_full = read_csv(data_paths[split]["human_labels"])
    machine_training_labels_full = read_csv(data_paths[split]["human_labels"])
    
    image_locations_partial = pandas.read_csv(data_paths[split]["image_locations"], nrows=100)
    human_training_labels_partial = pandas.read_csv(data_paths[split]["human_labels"], nrows=100)
    machine_training_labels_partial = pandas.read_csv(data_paths[split]["machine_labels"], nrows=100)
    print(class_descriptions.columns)
