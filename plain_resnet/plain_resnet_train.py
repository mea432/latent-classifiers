import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse

from tqdm import tqdm

from plain_resnet import PlainResNetClassifier


def train_plain_classifier(args):
    """
    Trains a standard ResNet-18 classifier directly on images.
    """
    device = torch.device("mps")
    print(f"Using device: {device}")

    # 1. Set up data loaders
    # The image size is 224x224, standard for ResNet models.
    # We also add normalization, which is a standard practice for models
    # trained on ImageNet-like data.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(args.dataset_path, "train")
    valid_dir = os.path.join(args.dataset_path, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(valid_dir):
        print(
            f"Error: 'train' or 'valid' directories not found in '{args.dataset_path}'."
        )
        print(
            "Please ensure your dataset is structured with 'train' and 'valid' subdirectories."
        )
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")

    # 2. Initialize the classifier model, optimizer, and loss function
    model = PlainResNetClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print("Plain classifier model initialized.")

    # 3. Start the training and validation loop
    print("\n--- Starting Plain Classifier Training ---")
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}"
        ):
            images, labels = images.to(device), labels.to(device)

            # Forward pass directly through the classifier
            optimizer.zero_grad()
            outputs = model(images)

            # Calculate loss and update weights
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        accuracy = 100 * correct / total

        print(f"--- End of Epoch [{epoch + 1}/{args.epochs}] ---")
        print(f"    Avg. Training Loss: {avg_train_loss:.4f}")
        print(f"    Avg. Validation Loss: {avg_valid_loss:.4f}")
        print(f"    Validation Accuracy: {accuracy:.2f}%")

        # Checkpoint the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "../checkpoints/best_plain_classifier.pth")
            print(f"    New best model saved with accuracy: {accuracy:.2f}%")

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy achieved: {best_accuracy:.2f}%")
    print("Best model saved to 'best_plain_classifier.pth'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a standard ResNet classifier on images."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Users/easonni/Desktop/Coding/science fair/FusedTensorImageClassifier/dataset",
        help="Path to the root of the dataset (containing 'train' and 'valid' subdirectories).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )

    args = parser.parse_args()
    train_plain_classifier(args)
