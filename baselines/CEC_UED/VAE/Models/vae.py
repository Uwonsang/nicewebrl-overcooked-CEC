import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import numpy as np
import optax

class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        
        # (B, 9, 9, 15)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),  # 9→9
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        
        # (B, 9, 9, 32)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(3, 3),  # 9→3
            padding='VALID',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # (B, 3, 3, 64)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),  # 3→3
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        
        # (B, 3, 3, 64)
        x = x.reshape((x.shape[0], -1))  # (B, 576)

        mean   = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        logvar = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return mean, logvar

class Decoder(nn.Module):
    output_channel: int  # 15

    @nn.compact
    def __call__(self, z):
        # (B, latent_dim)
        z = nn.Dense(3 * 3 * 64, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(z)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 3, 3, 64))
        
        # (B, 3, 3, 64)
        z = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),  # 3→3  ← 추가된 층
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z)
        
        # (B, 3, 3, 64)
        z = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),  # 3→3
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z)
        
        # (B, 3, 3, 32)
        z = nn.ConvTranspose(
            features=self.output_channel,
            kernel_size=(3, 3),
            strides=(3, 3),  # 3→9
            padding='VALID',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)

        # (B, 9, 9, 15)
        return z

class VAE(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, rng):
        # x: (B,9,9,26)
        mean, logvar = Encoder(self.config["latent_dim"])(x)
        std = jnp.exp(0.5 * logvar)

        rng, _rng = jax.random.split(rng)
        z = mean + std * jax.random.normal(_rng, mean.shape)

        recon = Decoder(self.config["output_channels"])(z) 

        return recon, mean, logvar

def vae_loss(params, apply_fn, x, rng, beta):
    logits, mean, logvar = apply_fn(params, x, rng)
    x_flat      = x.reshape((x.shape[0], -1))
    logits_flat = logits.reshape((logits.shape[0], -1))
    recon_loss = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits_flat, x_flat), axis=-1)) 
    kl_loss     = jnp.mean(-0.5 * jnp.sum(1.0 + logvar - mean**2 - jnp.exp(logvar), axis=-1))
    loss = recon_loss + beta * kl_loss
    return loss, (recon_loss, kl_loss)

class Encoder_crop(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # (B, 5, 5, C)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),  # 5→5
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # (B, 5, 5, 32)
        x = nn.Conv(
            features=64,
            kernel_size=(5, 5),
            strides=(5, 5),  # 5→1
            padding='VALID',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # (B, 1, 1, 64)
        x = x.reshape((x.shape[0], -1))  # (B, 64)

        mean   = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        logvar = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return mean, logvar


class Decoder_crop(nn.Module):
    output_channel: int

    @nn.compact
    def __call__(self, z):
        # (B, latent_dim)
        z = nn.Dense(1 * 1 * 64, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(z)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 1, 1, 64))

        # (B, 1, 1, 64)  →  (B, 5, 5, 32)
        # VALID: (1-1)*5 + 5 = 5 ✓
        z = nn.ConvTranspose(
            features=32,
            kernel_size=(5, 5),
            strides=(5, 5),
            padding='VALID',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z)

        # (B, 5, 5, 32)  →  (B, 5, 5, output_channel)
        z = nn.ConvTranspose(
            features=self.output_channel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)

        # (B, 5, 5, output_channel)
        return z


class VAE_crop(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, rng):
        # x: (B, 5, 5, C)
        mean, logvar = Encoder_crop(self.config["latent_dim"])(x)
        std = jnp.exp(0.5 * logvar)

        rng, _rng = jax.random.split(rng)
        z = mean + std * jax.random.normal(_rng, mean.shape)

        recon = Decoder_crop(self.config["output_channels"])(z)
        # recon: (B, 5, 5, output_channels)

        return recon, mean, logvar