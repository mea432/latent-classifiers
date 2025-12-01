import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from autoencoder.autoencoder import Autoencoder


def run_inference(args):
    """
    Loads a trained autoencoder model, passes an image through the encoder,
    prints the latent tensor, and saves the reconstructed image.
    """
    device = torch.device("mps")
    print(f"Using device: {device}")

    # 1. Load the trained autoencoder model
    print(f"Loading trained autoencoder model from {args.model_path}...")
    try:
        # Instantiate the full Autoencoder model to load the state dictionary
        model = Autoencoder(in_channels=3)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'")
        print("Please ensure you have a trained model saved from running train.py")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Extract the encoder and decoder parts
    encoder = model.encoder
    decoder = model.decoder

    # 2. Load and preprocess the input image
    print(f"Loading and preprocessing image: {args.image_path}")
    try:
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{args.image_path}'")
        return

    # The image size should match the size used during training
    image_size = 224
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    # Add a batch dimension (B, C, H, W) and send to the correct device
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Pass the image through the encoder
    print("\nEncoding image into latent space...")
    with torch.no_grad():
        latent_tensor = encoder(input_tensor)

    # 4. Print the latent-space tensor
    print("--- Latent Space Tensor ---")
    # Limiting the print output for large tensors
    torch.set_printoptions(profile="short")
    print(latent_tensor)
    print(f"\nShape of the latent tensor: {latent_tensor.shape}")
    print("---------------------------\n")
    torch.set_printoptions(profile="default")  # Reset print options

    # 5. Pass the latent tensor through the decoder to reconstruct
    print("Decoding latent space back into an image...")
    with torch.no_grad():
        reconstructed_tensor = decoder(latent_tensor)

    # 6. Save the reconstructed image
    if args.output_path:
        output_image_path = args.output_path
    else:
        # Create a default output path
        base, ext = os.path.splitext(os.path.basename(args.image_path))
        output_image_path = f"{base}_reconstructed.png"

    # Convert tensor to a PIL Image
    # Squeeze removes the batch dimension, cpu() moves it to the CPU
    reconstructed_image = transforms.ToPILImage()(reconstructed_tensor.squeeze(0).cpu())

    try:
        reconstructed_image.save(output_image_path)
        print(f"Successfully saved reconstructed image to: {output_image_path}")
    except Exception as e:
        print(f"Error saving the reconstructed image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pass an image through the autoencoder to get the latent tensor and reconstruction."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="checkpoints/best_autoencoder.pth",
        help="Path to the trained autoencoder model file (.pth).",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Optional: Path to save the reconstructed image.",
    )

    args = parser.parse_args()
    run_inference(args)
