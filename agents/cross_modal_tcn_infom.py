import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.infom import InFOMAgent, get_config as get_infom_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import Actor, IntentionEncoder, MLP, Value, VectorField, default_init


def _l2_normalize(x, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def symmetric_infonce_loss(h_state, h_rgb, temperature):
    """Symmetric in-batch InfoNCE for paired state/RGB embeddings."""
    h_state = _l2_normalize(h_state)
    h_rgb = _l2_normalize(h_rgb)
    similarities = h_state @ h_rgb.T
    logits = similarities / temperature
    labels = jnp.arange(h_state.shape[0])

    state_to_rgb = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    rgb_to_state = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
    loss = state_to_rgb + rgb_to_state

    eye = jnp.eye(h_state.shape[0], dtype=bool)
    negative_similarities = jnp.where(eye, jnp.nan, similarities)
    info = {
        'loss': loss,
        'state_to_rgb_loss': state_to_rgb,
        'rgb_to_state_loss': rgb_to_state,
        'retrieval_acc_state_to_rgb': (jnp.argmax(logits, axis=1) == labels).mean(),
        'retrieval_acc_rgb_to_state': (jnp.argmax(logits.T, axis=1) == labels).mean(),
        'positive_similarity': jnp.diag(similarities).mean(),
        'negative_similarity_mean': jnp.nanmean(negative_similarities),
        'negative_similarity_max': jnp.nanmax(negative_similarities),
    }
    return loss, info


class StateEmbeddingEncoder(nn.Module):
    """Trainable state encoder for Method B's shared contrastive latent."""

    shared_latent_dim: int
    hidden_dims: tuple
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations):
        return MLP(
            hidden_dims=(*self.hidden_dims, self.shared_latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(observations)


class RGBEmbeddingEncoder(nn.Module):
    """RGB encoder plus linear projection into the shared contrastive latent."""

    encoder: nn.Module
    shared_latent_dim: int

    @nn.compact
    def __call__(self, observations_rgb, train=True):
        features = self.encoder(observations_rgb, train=train)
        return nn.Dense(self.shared_latent_dim, kernel_init=default_init())(features)


class CrossModalTCNInFOMAgent(InFOMAgent):
    """TCN-style contrastive cross-modal InFOM.

    Method B learns a shared unit-normalized latent for paired low-dimensional
    state and RGB views. InFOM pretraining/fine-tuning consume RGB latents;
    evaluation consumes state latents.
    """

    state_mean: Any
    state_std: Any
    latent_min: Any
    latent_max: Any

    def normalize_state(self, observations):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        normalized = (observations - self.state_mean) / self.state_std
        return jnp.clip(normalized, -self.config['state_clip'], self.config['state_clip'])

    def encode_state(self, observations, params=None, freeze=False):
        h_state = self.network.select('state_encoder')(
            self.normalize_state(observations),
            params=params,
        )
        if self.config['normalize_embeddings']:
            h_state = _l2_normalize(h_state)
        if freeze:
            h_state = jax.lax.stop_gradient(h_state)
        return h_state

    def encode_rgb(self, observations_rgb, params=None, train=True, freeze=False):
        h_rgb = self.network.select('rgb_encoder')(
            observations_rgb,
            train=train,
            params=params,
        )
        if self.config['normalize_embeddings']:
            h_rgb = _l2_normalize(h_rgb)
        if freeze:
            h_rgb = jax.lax.stop_gradient(h_rgb)
        return h_rgb

    def _latent_batch(self, batch, grad_params=None, freeze_rgb=False):
        h_rgb = self.encode_rgb(
            batch['third_person_observations'],
            params=grad_params,
            freeze=freeze_rgb,
        )
        h_next_rgb = self.encode_rgb(
            batch['next_third_person_observations'],
            params=grad_params,
            freeze=freeze_rgb,
        )

        latent_batch = {
            'observations': h_rgb,
            'next_observations': h_next_rgb,
            'actions': batch['actions'],
            'next_actions': batch['next_actions'],
            'terminals': batch['terminals'],
            'observation_min': self.latent_min,
            'observation_max': self.latent_max,
        }
        if 'masks' in batch:
            latent_batch['masks'] = batch['masks']
        if 'rewards' in batch:
            latent_batch['rewards'] = batch['rewards']
        return latent_batch, h_rgb, h_next_rgb

    def _state_latents(self, batch, grad_params=None, freeze_state=False):
        h_state = self.encode_state(
            batch['observations'],
            params=grad_params,
            freeze=freeze_state,
        )
        h_next_state = self.encode_state(
            batch['next_observations'],
            params=grad_params,
            freeze=freeze_state,
        )
        return h_state, h_next_state

    def infonce_alignment_loss(self, batch, grad_params):
        h_state, h_next_state = self._state_latents(batch, grad_params=grad_params)
        _, h_rgb, h_next_rgb = self._latent_batch(batch, grad_params=grad_params)

        current_loss, current_info = symmetric_infonce_loss(
            h_state,
            h_rgb,
            self.config['infonce_temperature'],
        )
        next_loss, next_info = symmetric_infonce_loss(
            h_next_state,
            h_next_rgb,
            self.config['infonce_temperature'],
        )
        infonce_loss = 0.5 * (current_loss + next_loss)
        info = {
            'infonce/loss': infonce_loss,
            'infonce/current_loss': current_loss,
            'infonce/next_loss': next_loss,
            'infonce/temperature': self.config['infonce_temperature'],
            'embedding/rgb_norm': jnp.linalg.norm(h_rgb, axis=-1).mean(),
            'embedding/state_norm': jnp.linalg.norm(h_state, axis=-1).mean(),
            'embedding/rgb_std': h_rgb.std(),
            'embedding/state_std': h_state.std(),
        }
        for key, value in current_info.items():
            info[f'infonce/current_{key}'] = value
        for key, value in next_info.items():
            info[f'infonce/next_{key}'] = value
        return infonce_loss, info

    def state_behavioral_cloning_loss(self, batch, grad_params):
        h_state, _ = self._state_latents(batch, grad_params=grad_params)
        state_batch = {
            'observations': h_state,
            'actions': batch['actions'],
        }
        bc_loss, bc_info = self.behavioral_cloning_loss(state_batch, grad_params)
        return bc_loss, {f'bc_state/{k}': v for k, v in bc_info.items()}

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        rewards = batch['rewards']
        reward_preds = self.network.select('reward')(
            observations,
            params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()
        pred_centered = reward_preds - reward_preds.mean()
        reward_centered = rewards - rewards.mean()
        corr = (
            (pred_centered * reward_centered).mean()
            / (reward_preds.std() * rewards.std() + 1e-8)
        )

        return reward_loss, {
            'reward_loss': reward_loss,
            'pred_reward_mean': reward_preds.mean(),
            'pred_reward_std': reward_preds.std(),
            'true_reward_mean': rewards.mean(),
            'true_reward_std': rewards.std(),
            'pred_true_corr': corr,
        }

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, flow_occupancy_rng = jax.random.split(rng)

        latent_batch, h_rgb, _ = self._latent_batch(batch, grad_params=grad_params)

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            latent_batch,
            grad_params,
            flow_occupancy_rng,
        )
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        bc_rgb_loss, bc_rgb_info = self.behavioral_cloning_loss(latent_batch, grad_params)
        for k, v in bc_rgb_info.items():
            info[f'bc_rgb/{k}'] = v

        infonce_loss, infonce_info = self.infonce_alignment_loss(batch, grad_params)
        info.update(infonce_info)

        bc_state_loss, bc_state_info = self.state_behavioral_cloning_loss(batch, grad_params)
        info.update(bc_state_info)

        infom_pretraining_loss = flow_occupancy_loss + bc_rgb_loss
        joint_loss = (
            infom_pretraining_loss
            + self.config['lambda_infonce'] * infonce_loss
            + self.config['lambda_bc_state_extra'] * bc_state_loss
        )
        warmup_loss = self.config['lambda_infonce'] * infonce_loss
        in_warmup = self.network.step <= self.config['warmup_align_steps']
        loss = jnp.where(in_warmup, warmup_loss, joint_loss)

        h_state, _ = self._state_latents(batch, grad_params=grad_params)
        a_rgb = self.network.select('actor')(h_rgb, params=grad_params).mode()
        a_state = self.network.select('actor')(h_state, params=grad_params).mode()
        action_mse = jnp.square(a_rgb - a_state).mean()
        action_cos = (
            (a_rgb * a_state).sum(axis=-1)
            / (jnp.linalg.norm(a_rgb, axis=-1) * jnp.linalg.norm(a_state, axis=-1) + 1e-8)
        ).mean()

        info.update(
            {
                'loss/total_loss': loss,
                'loss/infom_pretraining_loss': infom_pretraining_loss,
                'loss/weighted_infonce_loss': self.config['lambda_infonce'] * infonce_loss,
                'loss/weighted_bc_state_extra_loss': self.config['lambda_bc_state_extra'] * bc_state_loss,
                'phase/is_alignment_warmup': in_warmup.astype(jnp.float32),
                'actor_modality/action_mse_rgb_state': action_mse,
                'actor_modality/action_cos_rgb_state': action_cos,
            }
        )
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetuning_loss(self, batch, grad_params, full_update=True, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, flow_occupancy_rng, actor_rng = jax.random.split(rng, 4)

        freeze_encoders = self.config['freeze_encoders_during_finetune']
        latent_batch, h_rgb, _ = self._latent_batch(
            batch,
            grad_params=grad_params,
            freeze_rgb=freeze_encoders,
        )

        reward_loss, reward_info = self.reward_loss(latent_batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        critic_loss, critic_info = self.critic_loss(latent_batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            latent_batch,
            grad_params,
            flow_occupancy_rng,
        )
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        if full_update:
            actor_loss, actor_info = self.actor_loss(latent_batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            actor_loss = 0.0

        h_state, _ = self._state_latents(batch, grad_params=grad_params, freeze_state=freeze_encoders)
        a_rgb = self.network.select('actor')(h_rgb, params=grad_params).mode()
        a_state = self.network.select('actor')(h_state, params=grad_params).mode()
        info['actor_modality/action_mse_rgb_state'] = jnp.square(a_rgb - a_state).mean()

        loss = reward_loss + critic_loss + flow_occupancy_loss + actor_loss
        info['loss/total_loss'] = loss
        return loss, info

    def _restore_encoder_params(self, new_network):
        params = new_network.params.copy()
        for module_name in ('modules_rgb_encoder', 'modules_state_encoder'):
            try:
                params[module_name] = self.network.params[module_name]
            except TypeError:
                params = params.copy({module_name: self.network.params[module_name]})
        return new_network.replace(params=params)

    @jax.jit
    def pretrain(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic_vf')
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.finetuning_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if self.config['freeze_encoders_during_finetune']:
            new_network = self._restore_encoder_params(new_network)
        self.target_update(new_network, 'critic_vf')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        state_latents = self.encode_state(observations)
        dist = self.network.select('actor')(state_latents, temperature=temperature)
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_third_person_observations=None,
        state_mean=None,
        state_std=None,
        latent_min=None,
        latent_max=None,
    ):
        if ex_third_person_observations is None:
            raise ValueError(
                'cross_modal_tcn_infom requires third_person_observations. '
                'Use a bridge dataset and enable RGB auxiliary frame stacking.'
            )
        if state_mean is None or state_std is None:
            raise ValueError(
                'cross_modal_tcn_infom requires fixed state normalizer statistics '
                '(state_mean, state_std).'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        shared_latent_dim = config['shared_latent_dim']
        latent_shape = (*ex_observations.shape[:-1], shared_latent_dim)
        ex_latent_observations = jnp.zeros(latent_shape, dtype=jnp.float32)
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        action_dim = ex_actions.shape[-1]

        rgb_encoder_module = encoder_modules[config['rgb_encoder']]
        rgb_encoder_def = RGBEmbeddingEncoder(
            encoder=rgb_encoder_module(),
            shared_latent_dim=shared_latent_dim,
        )
        state_encoder_def = StateEmbeddingEncoder(
            shared_latent_dim=shared_latent_dim,
            hidden_dims=config['state_encoder_hidden_dims'],
            layer_norm=config['state_encoder_layer_norm'],
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
        )
        intention_encoder_def = IntentionEncoder(
            hidden_dims=config['intention_encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['intention_encoder_layer_norm'],
        )
        critic_vf_def = VectorField(
            vector_dim=shared_latent_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
        )
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
        )

        network_info = dict(
            rgb_encoder=(rgb_encoder_def, (ex_third_person_observations,)),
            state_encoder=(state_encoder_def, (ex_observations,)),
            critic=(critic_def, (ex_latent_observations, ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_latent_observations,
                ex_times,
                ex_latent_observations,
                ex_actions,
                ex_latents,
            )),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_latent_observations,
                ex_times,
                ex_latent_observations,
                ex_actions,
                ex_latents,
            )),
            intention_encoder=(intention_encoder_def, (ex_latent_observations, ex_actions)),
            actor=(actor_def, (ex_latent_observations,)),
            reward=(reward_def, (ex_latent_observations,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params['modules_target_critic_vf'] = network.params['modules_critic_vf']

        if latent_min is None:
            latent_min = -jnp.ones((shared_latent_dim,), dtype=jnp.float32)
        if latent_max is None:
            latent_max = jnp.ones((shared_latent_dim,), dtype=jnp.float32)

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            state_mean=jnp.asarray(state_mean, dtype=jnp.float32),
            state_std=jnp.asarray(state_std, dtype=jnp.float32),
            latent_min=jnp.asarray(latent_min, dtype=jnp.float32),
            latent_max=jnp.asarray(latent_max, dtype=jnp.float32),
        )


def get_config():
    config = get_infom_config()
    config.agent_name = 'cross_modal_tcn_infom'
    config.encoder = None
    config.bridge_loss_weight = 0.0

    config.bridge_method = 'infonce'
    config.pretrain_view = 'paired_state_rgb'
    config.finetune_view = 'rgb_only'
    config.eval_view = 'state_only'
    config.state_encoder = 'mlp'
    config.state_encoder_hidden_dims = (512, 512, 512)
    config.state_encoder_layer_norm = True
    config.rgb_encoder = 'impala_small'
    config.rgb_frame_stack = 3
    config.rgb_p_aug = 0.5
    config.shared_latent_dim = 512
    config.normalize_embeddings = True
    config.infonce_temperature = 0.1
    config.lambda_infonce = 1.0
    config.state_clip = 5.0
    config.state_norm_eps = 1e-8
    config.warmup_align_steps = 10000
    config.lambda_bc_state_extra = 1.0
    config.freeze_encoders_during_finetune = True
    config.device_bridge_cache = False

    return ml_collections.ConfigDict(config)
