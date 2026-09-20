import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax
import wandb
import pickle

class NormalGaussianZ:
    def __init__(self, config):
        self.z_dim = config["z_dim"]
        self.z_list_size = config["z_list_size"]

    def get_z(self, key):
        return jax.random.normal(key, (self.z_list_size, self.z_dim))

class AdversaryNet(nn.Module):
    z_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(2 * self.z_dim)(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        return mean, log_std

class AdversarialZ:
    def __init__(self, config, lr_schedule, key):
        self.z_dim = config["ENV_KWARGS"]['vae_config']['latent_dim']
        self.batch_size = config["NUM_ENVS"]
        self.hidden_dim = config["gen_hidden_dim"]
        self.wandb_mode = config["WANDB_MODE"]
        self.kl_coeff = config["kl_coeff"] if config["use_kl"] else 0
        self.entropy_coeff = config["entropy_coeff"] if config["use_entropy"] else 0

        self.use_mean_clipping = config["use_mean_clipping"]
        self.clip_mean_min     = config.get('clip_mean_min', -0.5)
        self.clip_mean_max     = config.get('clip_mean_max',  0.5)
        self.use_std_scaling   = config["use_std_scaling"]
        self.scale_log_std     = config.get('scale_log_std',   1.0)
        self.scale_std_offset  = config.get('scale_std_offset', 0)
        self.z_list_size = config["z_list_size"]

        if config["filepath"] is not None:
            self.save_dir = os.path.join(config["filepath"], "models")
            os.makedirs(self.save_dir, exist_ok=True)

        self.net = AdversaryNet(z_dim=self.z_dim, hidden_dim=self.hidden_dim)
        params = self.net.init(key, jnp.zeros((1, self.z_dim)))
        tx = self._build_optimizer(config, lr_schedule)
        self.init_state = TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

    def _build_optimizer(self, config, lr_schedule):
        lr = lr_schedule if config["use_lr_scheduler"] else config["adversary_learning_rate"]
        wd = config["adversary_weight_decay"]

        transforms = [ ] 
        if config["use_grad_clip"]:
            transforms.append(optax.clip_by_global_norm(config["grad_clip_norm"]))
        transforms.append(optax.adamw(learning_rate=lr, weight_decay=wd))
        
        return optax.chain(*transforms)

    def _apply_net(self, apply_fn, params, z):
        mean, log_std = apply_fn(params, z)
        
        if self.use_mean_clipping:
            current_mean = jnp.mean(mean)
            shift = jnp.where(current_mean < self.clip_mean_min,  self.clip_mean_min - current_mean,
                    jnp.where(current_mean > self.clip_mean_max, self.clip_mean_max - current_mean, 0))
            mean = mean + shift

        if self.use_std_scaling:
            log_std = jnp.tanh(log_std) * self.scale_log_std + self.scale_std_offset
        
        return mean, log_std

    def get_z(self, z_gen_state, key):
        
        key, key1, key2 = jax.random.split(key, 3)
        z_prior = distrax.Normal(
            loc=jnp.zeros((self.z_list_size, self.z_dim,)),
            scale=jnp.ones((self.z_list_size, self.z_dim,)))

        z_sample_old = z_prior.sample(seed=key1) #(z_list_size, z_dim)

        mean, log_std = self._apply_net(
            z_gen_state.apply_fn, z_gen_state.params, z_sample_old) # (z_list_size, z_dim)
        std = jnp.exp(log_std) + 1e-6

        z_current = distrax.Normal(mean, std)
        z_sample_new = z_current.sample(seed=key2) #(z_list_size, z_dim)

        return z_sample_new, z_sample_old
        
    def compute_policy_stats(self, apply_fn, params, old_dist_diag=None, key=None):
        """Compute policy distribution stats after update."""

        z_normal = jax.random.normal(key, (self.batch_size, self.z_dim))
        mean, log_std = self._apply_net(apply_fn, params, z_normal)
        std = jnp.exp(log_std) + 1e-6

        new_dist = distrax.Normal(mean, std)
        entropy = new_dist.entropy().sum(axis=-1)

        kl_div = 0.0
        if old_dist_diag is not None:
            kl_div = self.kl_divergence(new_dist, old_dist_diag)

        return entropy, kl_div

    def train_step(self, z_gen_state, key, current_reward, trained_reward,
                         z_old, z_new, z_prior):

        # No grad (baseline)
        key, k_diag = jax.random.split(key)
        z_normal_diag = jax.random.normal(k_diag, (self.batch_size, self.z_dim))
        baseline_mean_diag, baseline_log_std_diag = self._apply_net(z_gen_state.apply_fn, z_gen_state.params, z_normal_diag)
        baseline_std_diag = jnp.exp(baseline_log_std_diag) + 1e-6
        baseline_dist_diag = distrax.Normal(baseline_mean_diag, baseline_std_diag)

        z_gen_apply_fn = z_gen_state.apply_fn
        def loss_fn(params):
            raw_regret = trained_reward - current_reward
            regret = (raw_regret - raw_regret.mean()) / (raw_regret.std() + 1e-8)

            mean, log_std = self._apply_net(z_gen_apply_fn, params, z_old)
            std = jnp.exp(log_std) + 1e-6
            new_dist = distrax.Normal(mean, std)
            log_probs = new_dist.log_prob(jax.lax.stop_gradient(z_new))
            weighted_log_probs = log_probs.sum(axis=-1) * regret
            
            kl_div = self.kl_divergence(new_dist, z_prior)
            entropy = new_dist.entropy().sum(axis=-1)

            loss = (
                -weighted_log_probs.mean()
                + self.kl_coeff * kl_div.mean()
                - self.entropy_coeff * entropy.mean()
            )
            
            return loss, (weighted_log_probs, kl_div, entropy, raw_regret)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(z_gen_state.params)
        new_z_gen_state = z_gen_state.apply_gradients(grads=grads)
        weighted_log_probs, kl_div, entropy, raw_regret = aux
        
        key, k_stats = jax.random.split(key)
        entropy_diag, kl_diag = self.compute_policy_stats(apply_fn=new_z_gen_state.apply_fn, params=new_z_gen_state.params, 
                                            old_dist_diag=baseline_dist_diag, key=k_stats)
        
        train_info = {
            "loss": loss,
            "episode_current_reward": current_reward.mean(),
            "episode_trained_reward": trained_reward.mean(),
            "regret": raw_regret.mean(),
            "weighted_log_probs": weighted_log_probs.mean(),
            "kl_divergence": kl_div.mean(),
            "policy_update": kl_diag.mean(),
            "entropy": entropy_diag.mean() 
            }

        return new_z_gen_state, train_info


    def log_train_info(self, metric):
        
        log_data = {
            'adversary/loss': metric['loss'],
            'adversary/episode_current_reward': metric['episode_current_reward'],
            'adversary/episode_trained_reward': metric['episode_trained_reward'],
            'adversary/regret': metric['regret'],
            'adversary/weighted_log_probs': metric['weighted_log_probs'],
            'adversary/kl_divergence': metric['kl_divergence'],
            'adversary/policy_update': metric['policy_update'],
            'adversary/entropy': metric['entropy'],
            'adversary/z_sample/mean': metric['z_sample/mean'],
            'adversary/z_sample/std': metric['z_sample/std'],
            'adversary/z_sample/log_prob': metric['log_prob']
        }
        
        if self.wandb_mode == "online":
            wandb.log(log_data)
 

    def save(self, z_gen_state_params, steps):
        path = os.path.join(self.save_dir, "adversary", f"adversary_{steps}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'params': jax.device_get(z_gen_state_params)}, f)

    def restore(self, z_gen_state, steps):
        path = os.path.join(self.save_dir, "adversary", f"adversary_{steps}.pkl")
        with open(path, 'rb') as f:
            params = pickle.load(f)
        return z_gen_state.replace(params=params)

    def kl_divergence(self, z_current, old_dist_diag):
        # KL(N1 || N0) analytic formula
        mean1 = z_current.loc
        std1  = z_current.scale
        mean0 = old_dist_diag.loc
        std0  = old_dist_diag.scale

        var1 = std1 ** 2
        var0 = std0 ** 2

        kl_per_dim = jnp.log(std0 / std1) + (var1 + (mean1 - mean0) ** 2) / (2.0 * var0) - 0.5
        kl_div = kl_per_dim.sum(axis=-1)
        
        return kl_div