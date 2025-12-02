import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

from tqdm import tqdm

from autoencoder.autoencoder import Autoencoder

# --- 1. Dataset Loading and Preprocessing ---


class OpenImagesDataset(Dataset):
    """
    A PyTorch Dataset to load images from a local directory.
    """

    def __init__(self, root_dir, image_size=224):
        """
        Args:
            root_dir (str): Directory with all the images.
            image_size (int): The size to which images will be resized.
        """
        self.root_dir = root_dir
        self.image_files = []
        # Supported image extensions
        self.extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

        print(f"Scanning for images in {root_dir}...")
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(self.extensions):
                self.image_files.append(os.path.join(root_dir, filename))

        if not self.image_files:
            raise FileNotFoundError(
                f"No image files found in {root_dir} with extensions {self.extensions}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        print(f"Found {len(self.image_files)} images. Transformations are set up.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image)


# --- 2. Training Setup ---


def train():
    """
    Main training function.
    """
    # --- Hyperparameters ---
    # With 80 training images and a batch size of 8, each epoch is 10 steps.
    # 20 epochs will be 200 steps. Increase this for a longer training session.
    num_epochs = 20
    batch_size = 8
    learning_rate = 1e-4
    image_size = 224
    dataset_dir = "autoencoder/autoencoder_dataset"

    # --- Setup Device, Model, Optimizer, and Loss ---
    device = torch.device("mps")
    print(f"Using device: {device}")

    model = Autoencoder(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- Prepare DataLoaders ---
    train_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "valid")

    if not os.path.isdir(train_dir) or not os.path.isdir(valid_dir):
        print(f"Error: Dataset directories not found in '{dataset_dir}'.")
        print("Please run prepare_dataset.py first to create the dataset.")
        return

    train_dataset = OpenImagesDataset(root_dir=train_dir, image_size=image_size)
    valid_dataset = OpenImagesDataset(root_dir=valid_dir, image_size=image_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("\n--- Starting Training ---")

    # --- 3. Training & Validation Loop ---
    best_valid_loss = float("inf")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for images in tqdm(
            train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"
        ):
            images = images.to(device)
            # Forward pass
            optimizer.zero_grad()
            reconstructed_images = model(images)
            loss = criterion(reconstructed_images, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images in valid_dataloader:
                images = images.to(device)
                reconstructed_images = model(images)
                loss = criterion(reconstructed_images, images)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_dataloader)

        print(f"--- End of Epoch [{epoch + 1}/{num_epochs}] ---")
        print(f"    Average Training Loss: {avg_train_loss:.4f}")
        print(f"    Average Validation Loss: {avg_valid_loss:.4f}")

        # --- 4. Checkpoint the best model ---
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), "checkpoints/best_autoencoder.pth")
            print(
                f"    New best model saved with validation loss: {best_valid_loss:.4f}"
            )

    print("\n--- Training Finished ---")
    print(f"Best validation loss achieved: {best_valid_loss:.4f}")
    print("Best model saved to 'checkpoints/best_autoencoder.pth'")


if __name__ == "__main__":
    train()
