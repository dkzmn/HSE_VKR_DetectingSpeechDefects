# -*- coding: utf-8 -*-
"""
VAE по мел-спектрограмме + классификатор по латентному вектору z (exp_18, Qi / Van hamme).
Encoder(mel) -> mu, logvar -> sample z -> Decoder(z) -> recon; Classifier(z) -> 2 класса.
"""
import torch
import torch.nn as nn


class MelVAE(nn.Module):
    def __init__(self, n_mels=80, n_frames=320, latent_dim=64, num_classes=2, n_letters=0):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.latent_dim = latent_dim
        self.n_letters = n_letters
        # Encoder: (1, 80, 320) -> down -> flat -> mu, logvar
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mels, n_frames)
            h = self.enc(x)
            self._enc_flat = h.view(1, -1).shape[1]
        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)
        # Decoder: z -> linear -> reshape -> conv transpose -> adaptive pool to (n_mels, n_frames)
        self.fc_dec = nn.Linear(latent_dim, self._enc_flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool2d((n_mels, n_frames)),
        )
        # Classifier по z
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim + n_letters, num_classes),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 5, 20)
        return self.dec(h)

    def forward(self, x, letters=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        if self.n_letters > 0 and letters is not None:
            z_cls = torch.cat([z, letters], dim=1)
        else:
            z_cls = z
        logits = self.classifier(z_cls)
        return recon, mu, logvar, logits


def get_model(n_mels=80, n_frames=320, latent_dim=64, num_classes=2, n_letters=0, **kwargs):
    return MelVAE(n_mels=n_mels, n_frames=n_frames, latent_dim=latent_dim, num_classes=num_classes, n_letters=n_letters, **kwargs)
