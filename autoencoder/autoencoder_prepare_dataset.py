import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(source_dir, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Splits a directory of images into train, validation, and test sets by
    copying them into a new directory structure.

    Args:
        source_dir (str): The directory containing the original images.
        output_dir (str): The root directory where the split dataset will be saved.
        split_ratio (tuple): A tuple with (train, valid, test) ratios. Must sum to 1.
        seed (int): A random seed for shuffling to ensure reproducibility.
    """
    if sum(split_ratio) != 1.0:
        raise ValueError("Split ratio must sum to 1.0")

    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Get all image file paths
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    print(f"Scanning for images in '{source_dir}'...")
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]

    if not image_files:
        print(
            f"Error: No images with supported extensions {extensions} found in '{source_dir}'"
        )
        return

    print(f"Found {len(image_files)} images.")

    # Shuffle the files for random splitting
    random.seed(seed)
    random.shuffle(image_files)

    # Calculate split indices
    num_images = len(image_files)
    train_end = int(num_images * split_ratio[0])
    valid_end = train_end + int(num_images * split_ratio[1])

    # Create file splits
    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]

    splits = {"train": train_files, "valid": valid_files, "test": test_files}

    # Create output directories and copy files
    print(f"Saving split dataset to '{output_dir}'...")
    for split_name, files in splits.items():
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copying {len(files)} files to '{split_dir}'...")
        for filename in files:
            shutil.copy(source_path / filename, split_dir / filename)

    print("\nDataset splitting complete.")
    print(f"  Training set: {len(train_files)} images")
    print(f"  Validation set: {len(valid_files)} images")
    print(f"  Test set: {len(test_files)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an image dataset into train, valid, and test sets."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/Users/easonni/fiftyone/open-images-v7/validation/data",
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="autoencoder_dataset",
        help="Directory to save the split dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling."
    )

    args = parser.parse_args()

    split_dataset(
        source_dir=args.source_dir, output_dir=args.output_dir, seed=args.seed
    )
