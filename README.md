# Latent Classifiers

This project explores the concept of using latent space representations from an autoencoder to train a classifier. It compares the performance of a standard ResNet classifier trained on images with a ResNet classifier trained on the latent space output of an autoencoder.

## Project Structure

```
├── autoencoder/
│   ├── autoencoder.py
│   ├── train.py
│   ├── autoencoder_download_dataset.py
│   ├── autoencoder_prepare_dataset.py
│   └── inference.py
├── autoencoder_latent_resnet/
│   ├── latent_resnet.py
│   ├── train.py
│   └── evaluate.py
├── plain_resnet/
│   ├── plain_resnet.py
│   ├── train.py
│   └── evaluate.py
├── checkpoints/
│   └── (empty)
└── README.md
```

- **`autoencoder/`**: Contains the autoencoder model, training scripts, and utilities for dataset management.
- **`autoencoder_latent_resnet/`**: Contains the ResNet model that classifies the latent space vectors from the autoencoder.
- **`plain_resnet/`**: Contains a standard ResNet model for baseline comparison, which is trained directly on images.
- **`checkpoints/`**: Directory to save the trained model weights.

## Getting Started

### Dependencies

This project requires Python 3 and the following libraries:

- `torch`
- `torchvision`
- `Pillow`
- `tqdm`

You can install these dependencies using pip:

```bash
pip install torch torchvision Pillow tqdm
```

### Dataset

The autoencoder is trained on the Open Images dataset. The classifiers are trained on a dataset of your choice, which should be structured as follows:

```
dataset/
├── train/
│   ├── class_a/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_b/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class_a/
    │   ├── image5.jpg
    │   └── image6.jpg
    └── class_b/
        ├── image7.jpg
        └── image8.jpg
```

## Running the Models

To ensure that Python can correctly resolve the imports between the different modules, you should run the training and evaluation scripts as modules from the root directory of the project using the `python -m` flag.

### 1. Autoencoder

**a) Download and Prepare the Dataset:**

First, download and prepare the dataset for the autoencoder.

```bash
python -m autoencoder.autoencoder_download_dataset
python -m autoencoder.autoencoder_prepare_dataset
```

**b) Train the Autoencoder:**

This will train the autoencoder and save the best model to `checkpoints/best_autoencoder.pth`.

```bash
python -m autoencoder.train
```

### 2. Plain ResNet Classifier

**a) Train the Classifier:**

This will train a standard ResNet classifier on your image dataset.

```bash
python -m plain_resnet.train --dataset_path /path/to/your/dataset
```

You can also specify other arguments like `--epochs`, `--batch_size`, and `--learning_rate`.

**b) Evaluate the Classifier:**

```bash
python -m plain_resnet.evaluate --dataset_path /path/to/your/dataset
```

### 3. Autoencoder Latent ResNet Classifier

**a) Train the Classifier:**

This will train the ResNet classifier on the latent space representations from the autoencoder.

```bash
python -m autoencoder_latent_resnet.train --dataset_path /path/to/your/dataset --autoencoder_path checkpoints/best_autoencoder.pth
```

You can also specify other arguments like `--epochs`, `--batch_size`, and `--learning_rate`.

**b) Evaluate the Classifier:**

```bash
python -m autoencoder_latent_resnet.evaluate --dataset_path /path/to/your/dataset --autoencoder_path checkpoints/best_autoencoder.pth
```
