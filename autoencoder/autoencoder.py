import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


class GDN(Module):
    """
    Generalized Divisive Normalization.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))

    Reference:
    "Density Modeling of Images using a Generalized Divisive Normalization"
    Johannes Ballé, Valero Laparra, Eero P. Simoncelli
    https://arxiv.org/abs/1511.06281

    This implementation is based on the one found in the paper's official repo:
    https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py
    and a PyTorch implementation:
    https://github.com/jorge-pessoa/pytorch-gdn
    """

    def __init__(
        self, ch, inverse=False, beta_min=1e-6, gamma_init=0.1, reparam_offset=2**-18
    ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = Parameter(gamma)

    def forward(self, x):
        unfold = False
        if x.dim() == 3:
            x = x.unsqueeze(-1)
            unfold = True

        _, ch, _, _ = x.size()

        # Beta bound and reparam
        beta = self.beta
        beta = F.softplus(beta) - self.beta_bound + self.beta_min

        # Gamma bound and reparam
        gamma = self.gamma
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        norm_pool = F.conv2d(x**2, gamma, beta, stride=1, padding=0)
        norm_pool = torch.sqrt(norm_pool)

        if self.inverse:
            x_out = x * norm_pool
        else:
            x_out = x / norm_pool

        if unfold:
            x_out = x_out.squeeze(-1)
        return x_out


class Encoder(nn.Module):
    """
    The Encoder part of the autoencoder.
    """

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=5, stride=2, padding=2),
            GDN(192),
            nn.Conv2d(192, 192, kernel_size=5, stride=2, padding=2),
            GDN(192),
            nn.Conv2d(192, 192, kernel_size=5, stride=2, padding=2),
            GDN(192),
            nn.Conv2d(192, 192, kernel_size=5, stride=2, padding=2),
            GDN(192),
        )

    def forward(self, x):
        return self.sequential(x)


class Decoder(nn.Module):
    """
    The Decoder part of the autoencoder.
    """

    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(
                192, 192, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            GDN(192, inverse=True),
            nn.ConvTranspose2d(
                192, 192, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            GDN(192, inverse=True),
            nn.ConvTranspose2d(
                192, 192, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            GDN(192, inverse=True),
            nn.ConvTranspose2d(
                192, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            GDN(out_channels, inverse=True),
        )

    def forward(self, x):
        return self.sequential(x)


class Autoencoder(nn.Module):
    """
    Autoencoder with convolutional layers and GDN/IGDN normalization.
    """

    def __init__(self, in_channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_channels=in_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Example usage:
    # Create an encoder and a decoder instance for 3-channel (RGB) images
    encoder = Encoder(in_channels=3)
    decoder = Decoder(out_channels=3)

    # Create a full autoencoder
    autoencoder = Autoencoder(in_channels=3)

    print("Encoder architecture:")
    print(encoder)
    print("\nDecoder architecture:")
    print(decoder)
    print("\nAutoencoder architecture:")
    print(autoencoder)

    # Create a dummy input tensor (e.g., a batch of 1 image, 3 channels, 224x224 pixels)
    input_tensor = torch.randn(1, 3, 224, 224)
    print(f"\nInput tensor shape: {input_tensor.shape}")

    # Pass the tensor through the encoder
    encoded_tensor = encoder(input_tensor)
    print(f"Encoded tensor shape: {encoded_tensor.shape}")

    # Pass the encoded tensor through the decoder
    decoded_tensor = decoder(encoded_tensor)
    print(f"Decoded tensor shape: {decoded_tensor.shape}")

    # Pass the tensor through the full autoencoder
    output_tensor = autoencoder(input_tensor)
    print(f"Autoencoder output tensor shape: {output_tensor.shape}")

    # Check if the output shape is the same as the input shape
    assert input_tensor.shape == decoded_tensor.shape, (
        "Decoder output shape does not match input shape!"
    )
    assert input_tensor.shape == output_tensor.shape, (
        "Autoencoder output shape does not match input shape!"
    )
    print("\nForward passes successful!")

    # You can now train this autoencoder like any other PyTorch model.
    # For example, using Mean Squared Error loss:
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    #
    # ... training loop ...
    # loss = criterion(output_tensor, input_tensor)
    # loss.backward()
    # optimizer.step()
