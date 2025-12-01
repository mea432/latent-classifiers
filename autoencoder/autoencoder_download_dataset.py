# 1. Install FiftyOne (if you haven't already)
# !pip install fiftyone

import fiftyone as fo
import fiftyone.zoo as foz

# Define the parameters for your download
dataset_name = "open-images-subset-1000"
split_to_use = "validation"  # or "train", "test"
number_of_samples = 1000

print(f"Starting download of {number_of_samples} images from Open Images V7...")

# Download the specified subset of the dataset
# You can customize `label_types`, `classes`, etc. if you have specific needs.
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split=split_to_use,
    max_samples=number_of_samples,
    shuffle=True,  # Randomly select the samples
    seed=51,  # For reproducibility
    dataset_name=dataset_name,
    # Set to [] if you only want the images and no annotations
    label_types=[],
)

print(f"Successfully downloaded {len(dataset)} samples.")
print(
    f"The images are stored at: {dataset.default_dir}"
)  # /Users/easonni/fiftyone/open-images-v7/validation
