import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np
import optax


class Encoder(nn.Module):
    code_dim: int

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(3, 3),
            padding='VALID',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.code_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        return x

class Decoder(nn.Module):
    output_channels: int
    
    @nn.compact
    def __call__(self, z):

        z = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',              
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z)

        z = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z)

        z = nn.ConvTranspose(
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=(3, 3),
            padding='VALID',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z) 
        
        return z

# VQ MODULE
class VectorQuantizer(nn.Module):
    code_dim: int
    num_codes: int
    beta_vq: float

    @nn.compact
    def __call__(self, z_e):
        # z_e: (B, H, W, D)
        codebook = self.param(
            'codebook',
            nn.initializers.uniform(scale=1.0 / self.num_codes),
            (self.num_codes, self.code_dim)
        )

        Batch, Height, Width, code_dim = z_e.shape
        z_flat = z_e.reshape(-1, code_dim)  # (BHW, D)

        # distances: ||x - e||^2 == x^2 + e^2 - 2 z e
        dist = (
            jnp.sum(z_flat ** 2, axis=1, keepdims=True)
            + jnp.sum(codebook ** 2, axis=1)
            - 2 * z_flat @ codebook.T
        )  # (BHW, K)

        indices = jnp.argmin(dist, axis=1)  # (BHW,)
        z_q = codebook[indices].reshape(z_e.shape) # (B, H, W, D)                                      # (N, D)

        codebook_loss = jnp.mean(
            jnp.sum(
                (jax.lax.stop_gradient(z_e) - z_q) ** 2,
                axis=(-1, -2, -3)  # H, W, D 합산
            )
        ) 

        commit_loss = self.beta_vq * jnp.mean(
            jnp.sum(
                (z_e - jax.lax.stop_gradient(z_q)) ** 2,
                axis=(-1, -2, -3)
            )
        )
        vq_loss = codebook_loss + commit_loss

        # Straight-through estimator
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        indices = indices.reshape(z_e.shape[0], z_e.shape[1], z_e.shape[2])  # (B,H,W)
        return z_q_st, vq_loss, indices

class VQVAE(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):
        z_e = Encoder(self.config["code_dim"])(x)  # (B, code_dim)

        z_q, vq_loss, indices = VectorQuantizer(self.config["code_dim"], self.config["num_codes"], self.config["beta_vq"])(z_e)

        logits = Decoder(self.config["output_channels"])(z_q)

        return logits, vq_loss, indices


def vqvae_loss(params, apply_fn, x):
    logits, vq_loss, _ = apply_fn(params, x)

    recon_loss = jnp.mean(
        jnp.sum(optax.sigmoid_binary_cross_entropy(
            logits.reshape(logits.shape[0], -1),
            x.reshape(x.shape[0], -1).astype(jnp.float32)
        ), axis=1)
    )
    loss = recon_loss + vq_loss

    return loss, (recon_loss, vq_loss)


