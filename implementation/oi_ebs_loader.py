from torch.utils.data import Dataset


class OpenImagesEBSDataset(Dataset):
    """A Torch wrapper for the OpenImages V6 dataset as stored on the file-tree (in this case an EBS volume). Allows
       for writing to disk in batches and iterating over those batches at network train/test time."""
    
    def __init__(self):
        # TODO: specify train, validation, or test
        # TODO: specify batch size in bytes
        # TODO: specify disk locations from config
        pass
    
    def add_item(self, item):
        """Saves an image and its associated data to disk, managing batch sizes while doing so."""
    
    def __len__(self):
        """Returns the number of entries in the OpenImages dataset. Required for a custom Torch Dataset."""
        pass
    
    def __getitem__(self, index):
        """Retrieves the entry at the given zero-index. Required for a custom Torch Dataset."""
        pass
