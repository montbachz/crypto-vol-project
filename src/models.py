from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_channels: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        flat_dim = hidden_channels * 6 * 7

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

        self.hidden_channels = hidden_channels

    def encode(self, x_masked: torch.Tensor, mask: torch.Tensor):
        # x_masked and mask are shape: (batch, 1, 6, 7)
        inp = torch.cat([x_masked, mask], dim=1)
        h = self.encoder(inp)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor):
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_channels, 6, 7)
        return self.decoder(h)

    def forward(self, x_masked: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(x_masked, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x_true: torch.Tensor,
    obs_mask: torch.Tensor,
    hidden_weight: float = 1.0,
    observed_weight: float = 0.1,
    beta: float = 1e-3,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
):
    hidden_mask = 1.0 - obs_mask

    hidden_mse = ((recon - x_true) ** 2 * hidden_mask).sum() / hidden_mask.sum().clamp_min(1.0)
    observed_mse = ((recon - x_true) ** 2 * obs_mask).sum() / obs_mask.sum().clamp_min(1.0)

    loss = hidden_weight * hidden_mse + observed_weight * observed_mse

    if mu is not None and logvar is not None:
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + beta * kl
    else:
        kl = torch.tensor(0.0, device=recon.device)

    return loss, hidden_mse, observed_mse, kl