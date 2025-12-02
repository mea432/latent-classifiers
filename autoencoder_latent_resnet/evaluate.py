import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import time

from autoencoder.autoencoder import Autoencoder
from autoencoder_latent_resnet.latent_resnet import LatentResNetClassifier

image_size = 224


def evaluate_classifier(args):
    """
    Evaluates the ResNetClassifier on the test set.
    """
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' not found.")
        return

    device = torch.device("mps")
    print(f"Using device: {device}")

    # 1. Load the pre-trained and frozen encoder
    print(f"Loading pre-trained encoder from '{args.autoencoder_path}'...")
    try:
        autoencoder_model = Autoencoder(in_channels=3)
        autoencoder_model.load_state_dict(
            torch.load(args.autoencoder_path, map_location=device)
        )
        encoder = autoencoder_model.encoder.to(device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    except FileNotFoundError:
        print(f"Error: Autoencoder model not found at '{args.autoencoder_path}'")
        return

    # 2. Load the trained classifier
    print(f"Loading trained classifier from '{args.classifier_path}'...")
    try:
        # Load the state dictionary first to inspect it
        state_dict = torch.load(args.classifier_path, map_location=device)

        # Infer num_classes from the final layer's shape in the state_dict
        # The key for the final layer's weights is 'resnet.fc.weight'
        num_classes = state_dict["resnet.fc.weight"].shape[0]
        print(f"Inferred {num_classes} classes from the model checkpoint.")

        # Initialize the model with the correct number of classes
        classifier = LatentResNetClassifier(
            num_classes=num_classes, in_channels=192
        ).to(device)

        # Load the state dictionary into the correctly sized model
        classifier.load_state_dict(state_dict)
        classifier.eval()
    except FileNotFoundError:
        print(f"Error: Classifier model not found at '{args.classifier_path}'")
        return
    except KeyError:
        print(
            "Error: Could not find 'resnet.fc.weight' in the model file. The model architecture may have changed."
        )
        return

    # 3. Set up the test data loader
    test_dir = os.path.join(args.dataset_path, "test")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"Found {len(test_dataset)} images in the test set.")

    # 4. Evaluate the model
    print("\n--- Starting Evaluation on Test Set ---")
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Start timer
            # start_time = time.time()

            # Pass through the two-stage model
            latent_tensors = encoder(images)

            start_time = time.time()
            outputs = classifier(latent_tensors)

            # End timer
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 5. Report results
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)

    print("\n--- Evaluation Results ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Time per Batch: {avg_inference_time * 1000:.4f} ms")
    print(
        f"Average Inference Time per Image: {avg_inference_time / args.batch_size * 1000:.4f} ms"
    )
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the latent space classifier on the test set."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Users/easonni/Desktop/Coding/science fair/FusedTensorImageClassifier/dataset",
        help="Path to the root of the dataset (containing a 'test' subdirectory).",
    )
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default="checkpoints/best_autoencoder.pth",
        help="Path to the best trained autoencoder model file (.pth).",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="checkpoints/best_latent_resnet_classifier.pth",
        help="Path to the best trained latent space classifier model file (.pth).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation."
    )

    args = parser.parse_args()
    evaluate_classifier(args)
