import torch
import torch.nn as nn
from torchvision.models import resnet18


class LatentResNetClassifier(nn.Module):
    """
    A ResNet-18 based classifier adapted to take the output of the
    autoencoder's encoder (a latent space tensor) as input.

    The standard ResNet-18 architecture is modified to:
    1. Accept an input with a custom number of channels (e.g., 192).
    2. Handle small input feature map sizes (e.g., 16x16) by replacing
       the initial aggressive max-pooling layer.
    3. Output a specified number of classes.
    """

    def __init__(self, num_classes=10, in_channels=192):
        """
        Args:
            num_classes (int): The number of output classes for the final layer.
            in_channels (int): The number of channels in the input tensor.
                               This should match the output channels of the encoder.
        """
        super(LatentResNetClassifier, self).__init__()

        # Load a ResNet-18 architecture without pre-trained weights
        self.resnet = resnet18(weights=None)

        # 1. Modify the first convolutional layer to accept `in_channels`.
        # The original layer is Conv2d(3, 64, ...).
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 2. Adapt for small input resolutions (e.g., 14x14).
        # The original maxpool layer (kernel_size=3, stride=2) is too aggressive
        # for a 14x14 feature map. We replace it with an identity layer,
        # effectively removing it to preserve dimensions.
        self.resnet.maxpool = nn.Identity()

        # 3. Modify the final fully connected layer for the desired number of classes.
        # The original layer is Linear(in_features=512, out_features=1000).
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass for the classifier.
        Args:
            x (torch.Tensor): The input tensor from the encoder, expected to have
                              a shape of (Batch, in_channels, H, W),
                              e.g., (4, 192, 14, 14).
        """
        return self.resnet(x)


if __name__ == "__main__":
    # --- Example Usage ---

    # This dummy tensor simulates the output of our autoencoder's encoder
    # It has a batch size of 4, 192 channels, and a 14x14 resolution.
    latent_space_tensor = torch.randn(4, 192, 14, 14)

    # Create the classifier for a 10-class problem
    classifier = LatentResNetClassifier(num_classes=10, in_channels=192)

    print("--- Custom ResNet-18 Classifier Architecture ---")
    print(classifier)
    print("\n")

    # Pass the latent space tensor through the classifier to get logits
    output_logits = classifier(latent_space_tensor)

    print("--- Example Forward Pass ---")
    print(f"Shape of the input (latent space tensor): {latent_space_tensor.shape}")
    print(f"Shape of the output (logits): {output_logits.shape}")

    # Verify the output shape is correct for a batch of 4 and 10 classes
    assert output_logits.shape == (4, 10), "Output shape is incorrect!"
    print("\nForward pass successful!")
