"""Implementation of a Vision Transformer for image classification.
See Dosovitskiy et al. 2020 Sections 3 and 4 for the details included here.
"""
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score

from config import BASE_DIR


class VisionTransformer(torch.nn.Module):

    def __init__(self, h, w, c, num_classes):
        super().__init__()
        
        # Data dimensions
        self.H = h  # Height of an image in pixels
        self.W = w  # Width of an image in pixels
        self.C = c  # Number of channels of the image (e.g. RGB=3)
        self.num_classes = num_classes  # Number of possible classes in the output data
        
        # Model parameters
        # See Table 1 of Dosovitsky et al. for other options
        self.image_patch_size = 4  # Patch resolution
        self.num_patches = int((self.H * self.W) / (self.image_patch_size ** 2))  # Check N
        self.transformer_hidden_size = 144  # Embedding dimensions throughout the Transformer
        self.transformer_num_layers = 12  # Number of Transformer encoder layers
        self.transformer_num_heads = 12  # Number of heads for multi-headed self attention in the Transformer
        self.transformer_MLP_size = 512  # dimension of feed-forward layer in the Transformer
        self.transformer_activation = "gelu"

        # Model layers
        # Patching and embedding: Done at once in the original implementation
        self.patch_embedding_layer = torch.nn.Conv2d(in_channels=self.C, out_channels=self.transformer_hidden_size,
                                                     kernel_size=(self.image_patch_size, self.image_patch_size),
                                                     stride=(self.image_patch_size, self.image_patch_size))
        
        # Position embedding layer
        self.position_embedding_layer = torch.nn.Embedding(num_embeddings=self.num_patches+1,
                                                           embedding_dim=self.transformer_hidden_size)
        
        # Transformer itself: out-of-the-box this isn't structured exactly like the original ViT implementation
        # TODO explain those differences
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.transformer_hidden_size,
                                                              nhead=self.transformer_num_heads, dropout=0,
                                                              dim_feedforward=self.transformer_MLP_size,
                                                              activation=self.transformer_activation)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                   num_layers=self.transformer_num_layers)
        
        # The final classification head: "one hidden layer at pre-training time, linear layer only at fine-tuning time"
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.transformer_hidden_size, out_features=self.transformer_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=self.transformer_hidden_size, out_features=self.num_classes),
            torch.nn.Softmax(dim=2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch the images, flatten, and linearly embed: all in one step!
        x_patches = self.patch_embedding_layer(x)
        # Flatten down to a list of patches instead of a grid
        x_patches = x_patches.flatten(start_dim=2)
        assert x_patches.shape[2] == self.num_patches
        
        # Zero initialize a class token Tensor
        class_token = torch.zeros(*(1, self.transformer_hidden_size, 1))
        class_token = class_token.repeat(*(batch_size, 1, 1))
        # Concatenate to the beginning of the patch sequence
        x_patches_with_class = torch.cat((class_token, x_patches), dim=2)
        assert x_patches_with_class.shape[2] == self.num_patches + 1
        
        # Add position embeddings to the sequence
        position_embeddings = self.position_embedding_layer(torch.arange(end=self.num_patches+1))
        position_embeddings = torch.transpose(*(position_embeddings, 0, 1))
        x_patches_with_class_and_posn = x_patches_with_class + position_embeddings
        assert x_patches_with_class_and_posn.shape[1] == self.transformer_hidden_size
        
        # Pass the data through the Transformer encoder
        x_patches_with_class_and_posn = x_patches_with_class_and_posn.permute(*(2, 0, 1))
        transformer_out = self.encoder(x_patches_with_class_and_posn)
        
        # Take from the Transformer's output the learned class token only
        transformer_out_class_token = transformer_out[0, :, :].unsqueeze(0)
        
        # Note: Since we're using PyTorch's built-in Transformer encoder an additional LayerNorm is not necessary here
        # Pass it through the classification head
        out = self.classification_head(transformer_out_class_token)
        assert out.shape[2] == 10
        return out


# Taken from the in-class exercise on 2020-10-15 covering image classification with CNNs
def train_model(optimizer, model, train_loader, device, num_epochs=5):
    losses = []
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
    
            # forwards
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            # backwards
            optimizer.zero_grad()
            loss.backward()
    
            # update params
            optimizer.step()
    
            if (i + 1) % 100 == 0:
                losses.append(loss.item())
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    return losses


# Taken from the in-class exercise on 2020-10-15 covering image classification with CNNs
def test_model(model, test_loader, device):
    # switch to `eval` model
    model.eval()

    y, y_hat = [], []

    with torch.no_grad():
        for X_i, y_i in test_loader:
            X_i, y_i = X_i.to(device), y_i.to(device)
            y_hat_i = model(X_i)
            y.extend(y_i.detach().cpu().tolist())
            discrete_preds = y_hat_i.argmax(dim=1).detach().cpu()
            y_hat.extend(discrete_preds)

    return y, y_hat


def save_image_from_array(img, path, mode="RGB"):
    img = Image.fromarray(img, mode)
    img.save(path)


if __name__ == "__main__":
    
    # Global parameters
    this_device = "cpu"
    num_epochs = 10
    batch_size = 4096
    adam_learning_rate = 0.0005
    adam_betas = (0.9, 0.999)
    
    # Load CIFAR-10
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root=BASE_DIR / "data" / "cifar_10_train",
                                                         train=True, download=True,
                                                         transform=torchvision.transforms.ToTensor())
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root=BASE_DIR / "data" / "cifar_10_test",
                                                        train=False, download=True,
                                                        transform=torchvision.transforms.ToTensor())
    
    # Make data loaders
    cifar10_train_data_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True,
                                                            num_workers=4)
    cifar10_test_data_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=True,
                                                           num_workers=4)
    
    # Get image resolution, num channels
    assert cifar10_train_dataset.data.shape[1:] == cifar10_test_dataset.data.shape[1:], \
        "Train and test image dimensions don't match!"
    N, H, W, C = cifar10_train_dataset.data.shape
    assert len(cifar10_train_dataset.classes) == len(cifar10_test_dataset.classes), \
        "Train and test number of classes don't match!"  # Sanity check but not strictly necessary
    cifar10_num_classes = len(cifar10_train_dataset.classes)
    
    ViT_model = VisionTransformer(h=H, w=W, c=C, num_classes=cifar10_num_classes)
    ViT_model.to(this_device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=ViT_model.parameters(), lr=adam_learning_rate, betas=adam_betas)
    
    training_losses = train_model(optimizer, ViT_model, cifar10_train_data_loader, this_device, num_epochs=num_epochs)
    y, y_hat = test_model(ViT_model, cifar10_test_data_loader, this_device)
    test_accuracy = accuracy_score(y, y_hat)
    print(f'Test accuracy: {round(test_accuracy, 4)}')
    
    # TODO make some graphs? plot the classification matrix? Generate other scores?
