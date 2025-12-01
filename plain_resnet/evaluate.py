import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import time

from plain_resnet.plain_resnet import PlainResNetClassifier


def evaluate_plain_classifier(args):
    """
    Evaluates the PlainResNetClassifier on the test set.
    """
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' not found.")
        return

    device = torch.device("mps")
    print(f"Using device: {device}")

    # 1. Load the trained classifier
    print(f"Loading trained plain classifier from '{args.classifier_path}'...")
    try:
        # Load the state dictionary first to inspect it
        state_dict = torch.load(args.classifier_path, map_location=device)

        # Infer num_classes from the final layer's shape in the state_dict
        num_classes = state_dict["resnet.fc.weight"].shape[0]
        print(f"Inferred {num_classes} classes from the model checkpoint.")

        # Initialize the model with the correct number of classes
        model = PlainResNetClassifier(num_classes=num_classes).to(device)

        # Load the state dictionary into the correctly sized model
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        print(f"Error: Classifier model not found at '{args.classifier_path}'")
        return
    except KeyError:
        print(
            "Error: Could not find 'resnet.fc.weight' in the model file. The model architecture may have changed."
        )
        return

    # 2. Set up the test data loader
    test_dir = os.path.join(args.dataset_path, "test")
    # Transformations must match the training setup for this model
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"Found {len(test_dataset)} images in the test set.")

    # 3. Evaluate the model
    print("\n--- Starting Evaluation on Test Set ---")
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Start timer
            start_time = time.time()

            # Pass directly through the classifier
            outputs = model(images)

            # End timer
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 4. Report results
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)

    print("\n--- Evaluation Results ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Time per Batch: {avg_inference_time * 1000:.4f} ms")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the plain ResNet classifier on the test set."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Users/easonni/Desktop/Coding/science fair/FusedTensorImageClassifier/dataset",
        help="Path to the root of the dataset (containing a 'test' subdirectory).",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="checkpoints/best_plain_classifier.pth",
        help="Path to the best trained plain classifier model file (.pth).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation."
    )

    args = parser.parse_args()
    evaluate_plain_classifier(args)
