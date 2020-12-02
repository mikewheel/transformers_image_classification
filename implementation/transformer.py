"""Right now this is really just scratch space...
See Dosovitskiy et al. 2020 Sections 3 and 4 for the details included here.
"""
from PIL import Image
import torch
import torchvision

from config import BASE_DIR

# Load CIFAR-10
# FIXME: do I need to add in a transform here to resize? Or center crop?
cifar10_train_dataset = torchvision.datasets.CIFAR10(root=BASE_DIR / "data" / "cifar_10_train",
                                                     train=True, download=True)
cifar10_test_dataset = torchvision.datasets.CIFAR10(root=BASE_DIR / "data" / "cifar_10_test",
                                                    train=False, download=True)

# Make data loaders
cifar10_train_data_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=4096, shuffle=True,
                                                        num_workers=4)
cifar10_test_data_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=4096, shuffle=True,
                                                       num_workers=4)

# View an image to verify its contents
img = Image.fromarray(cifar10_train_dataset.data[0, :, :, :], "RGB")
img.save(BASE_DIR / 'data' / 'example.png')

# Get image resolution, num channels
N_train, H_train, W_train, C_train = cifar10_train_dataset.data.shape
N_test, H_test, W_test, C_test = cifar10_test_dataset.data.shape
assert H_train == H_test and W_train == W_test and C_train == C_test, "Train and test image dimensions don't match!"

H, W, C = H_train, W_train, C_train
del H_train, W_train, C_train, H_test, W_test, C_test

P = 16  # Patch resolution

# See Table 1 of Dosovitsky et al. for other options
D = 768  # Embedding dimensions throughout the Transformer
L = 12  # Number of Transformer encoder layers
n_heads = 12  # Number of heads for multi-headed self attention in the Transformer
transformer_MLP_size = 3072  # dimension of feed-forward layer in the Transformer

# TODO: walk through components in the paper alongside the JAX implementation
# Split image into patches
# Linearly embed patches
# Pre-pend the "[class]" embedding (how?)
# Add position embeddings
# Transformer encoder
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=D, nhead=n_heads, dropout=0,
                                                 dim_feedforward=transformer_MLP_size, activation="gelu")
encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=L)
# Classifier

# TODO: train, test, and report results

