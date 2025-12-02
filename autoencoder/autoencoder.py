import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


class GDN(nn.Module):
    """
    Official GDN/IGDN from Ballé/Minnen (tensorflow/compression version).
    Uses:
      - lower-triangular reparam for gamma
      - softplus for positivity
      - pedestal offset for stability
    """

    def __init__(
        self,
        channels,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18,
    ):
        super().__init__()

        self.inverse = inverse
        self.channels = channels
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        # Pedestal ensures strictly positive pre-activation
        pedestal = reparam_offset**2
        self.pedestal = pedestal

        # Reparameterized beta: stored as sqrt form
        beta = torch.sqrt(torch.ones(channels) + pedestal)
        self.beta = nn.Parameter(beta)

        # Reparameterized gamma: lower triangular sqrt matrix
        # gamma_init * I but in sqrt-space
        eye = torch.eye(channels)
        g = gamma_init * eye + pedestal
        self.gamma = nn.Parameter(torch.sqrt(g))

        # Mask for lower-triangular only
        self.register_buffer("gamma_mask", torch.tril(torch.ones(channels, channels)))

    def forward(self, x):
        # ensure positivity using softplus
        beta = F.softplus(self.beta) ** 2 - self.pedestal
        beta = torch.clamp(beta, min=self.beta_min)

        # gamma: softplus squared -> full matrix -> mask to lower-triangular
        gamma = F.softplus(self.gamma) ** 2 - self.pedestal
        gamma = gamma * self.gamma_mask  # enforce lower-triangular

        # reshape for 1x1 conv
        gamma = gamma.view(self.channels, self.channels, 1, 1)

        # compute normalization denominator
        norm = F.conv2d(x**2, gamma, beta)
        norm = torch.sqrt(norm)

        if self.inverse:
            return x * norm
        else:
            return x / norm


class IGDN(GDN):
    """Inverse GDN is just GDN with inverse=True."""

    def __init__(self, channels, **kwargs):
        super().__init__(channels, inverse=True, **kwargs)


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, 192, 5, stride=2, padding=2),
            GDN(192),
            nn.Conv2d(192, 192, 5, stride=2, padding=2),
            GDN(192),
            nn.Conv2d(192, 192, 5, stride=2, padding=2),
            GDN(192),
            nn.Conv2d(192, 192, 5, stride=2, padding=2),
            # final layer: NO GDN (linear)
        )

    def forward(self, x):
        return self.sequential(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(192, 192, 5, stride=2, padding=2, output_padding=1),
            GDN(192, inverse=True),
            nn.ConvTranspose2d(192, 192, 5, stride=2, padding=2, output_padding=1),
            GDN(192, inverse=True),
            nn.ConvTranspose2d(192, 192, 5, stride=2, padding=2, output_padding=1),
            GDN(192, inverse=True),
            nn.ConvTranspose2d(
                192, out_channels, 5, stride=2, padding=2, output_padding=1
            ),
            # final layer: NO IGDN (linear)
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
