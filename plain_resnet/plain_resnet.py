import torch
import torch.nn as nn
from torchvision.models import resnet18

class PlainResNetClassifier(nn.Module):
    """
    A standard ResNet-18 classifier adapted to take raw images as input.

    This classifier is designed for direct image classification, using 224x224
    pixel dimensions as input, typical for ResNet-18.
    """
    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): The number of output classes for the final layer.
        """
        super(PlainResNetClassifier, self).__init__()
        
        # Load a ResNet-18 architecture without pre-trained weights
        # It's designed for 3 input channels and 224x224 images by default.
        self.resnet = resnet18(weights=None)

        # The first convolutional layer (conv1) and maxpool layer are kept as default
        # because the input is a standard 3-channel image of 224x224 resolution.

        # Modify the final fully connected layer for the desired number of classes
        # The original layer is Linear(in_features=512, out_features=1000).
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass for the classifier.
        Args:
            x (torch.Tensor): The input image tensor, expected to have
                              a shape of (Batch, 3, 224, 224).
        """
        return self.resnet(x)

if __name__ == '__main__':
    # --- Example Usage ---
    
    # This dummy tensor simulates a batch of 4 RGB images with 224x224 resolution.
    input_image_tensor = torch.randn(4, 3, 224, 224)
    
    # Create the classifier for a 10-class problem
    classifier = PlainResNetClassifier(num_classes=10)
    
    print("--- Plain ResNet-18 Classifier Architecture ---")
    print(classifier)
    print("\n")
    
    # Pass the input image tensor through the classifier
    output_logits = classifier(input_image_tensor)
    
    print("--- Example Forward Pass ---")
    print(f"Shape of the input (image tensor): {input_image_tensor.shape}")
    print(f"Shape of the output (logits): {output_logits.shape}")
    
    # Verify the output shape is correct for a batch of 4 and 10 classes
    assert output_logits.shape == (4, 10), "Output shape is incorrect!"
    print("\nForward pass successful!")
