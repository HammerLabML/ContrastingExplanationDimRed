from typing import Sequence
from flax import linen as nn


class MyAE(nn.Module):
    features: Sequence[int]
    input_dim: int

    def setup(self):
        self.encoder_layers = [nn.Dense(feat) for feat in self.features]
        self.decoder_layers = [nn.Dense(feat) for feat in list(reversed(self.features)) + [self.input_dim]]

    def encoder(self, inputs):
        x = inputs
        for lyr in self.encoder_layers:
            x = lyr(x)
            x = nn.relu(x)
        return x

    def __call__(self, inputs):
        x = inputs
        layers = self.encoder_layers + self.decoder_layers
        for i, lyr in enumerate(layers):
            x = lyr(x)
            if i != len(layers) - 1:
                x = nn.relu(x)
        return x
