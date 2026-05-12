import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


def augment(batch, keys, new_key_prefix=''):
    """Apply image augmentation to the given keys."""
    padding = 3
    batch_size = len(batch[keys[0]])
    crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
    crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
    for key in keys:
            batch[new_key_prefix + key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


def _stack_device_rgb(data, initial_locs, idxs, key, frame_stack):
    """Stack current and next RGB frames on device for fixed transition indices."""
    next_key = f'next_{key}'
    initial_idxs = initial_locs[jnp.searchsorted(initial_locs, idxs, side='right') - 1]

    obs = []
    next_obs = []
    for i in reversed(range(frame_stack)):
        cur_idxs = jnp.maximum(idxs - i, initial_idxs)
        obs.append(data[key][cur_idxs])
        if i != frame_stack - 1:
            next_obs.append(data[key][cur_idxs])
    next_obs.append(data[next_key][idxs])

    return jnp.concatenate(obs, axis=-1), jnp.concatenate(next_obs, axis=-1)


def _augment_device_rgb(batch, keys, rng, p_aug):
    """Apply the same random crop augmentation as the CPU path, but on device."""
    if p_aug is None or p_aug <= 0.0:
        return batch

    padding = 3
    batch_size = batch[keys[0]].shape[0]
    rng_apply, rng_crop = jax.random.split(rng)
    crop_froms = jax.random.randint(
        rng_crop,
        (batch_size, 2),
        minval=0,
        maxval=2 * padding + 1,
    )
    crop_froms = jnp.concatenate(
        [crop_froms, jnp.zeros((batch_size, 1), dtype=crop_froms.dtype)],
        axis=1,
    )
    apply_aug = jax.random.bernoulli(rng_apply, p_aug)

    for key in keys:
        cropped = batched_random_crop(batch[key], crop_froms, padding)
        if p_aug >= 1.0:
            batch[key] = cropped
        else:
            batch[key] = jax.lax.cond(
                apply_aug,
                lambda _: cropped,
                lambda _: batch[key],
                operand=None,
            )
    return batch


def _build_device_bridge_batch(data, initial_locs, observation_min, observation_max,
                               idxs, rng, rgb_frame_stack, p_aug):
    batch = {
        key: value[idxs]
        for key, value in data.items()
        if key not in ('third_person_observations', 'next_third_person_observations')
    }
    rgb, next_rgb = _stack_device_rgb(
        data,
        initial_locs,
        idxs,
        key='third_person_observations',
        frame_stack=rgb_frame_stack,
    )
    batch['third_person_observations'] = rgb
    batch['next_third_person_observations'] = next_rgb
    batch['observation_min'] = observation_min
    batch['observation_max'] = observation_max
    return _augment_device_rgb(
        batch,
        ('third_person_observations', 'next_third_person_observations'),
        rng,
        p_aug,
    )


@partial(jax.jit, static_argnames=('rgb_frame_stack', 'p_aug'))
def _device_bridge_batch_from_idxs(data, initial_locs, observation_min, observation_max,
                                   idxs, rng, rgb_frame_stack, p_aug):
    return _build_device_bridge_batch(
        data,
        initial_locs,
        observation_min,
        observation_max,
        idxs,
        rng,
        rgb_frame_stack,
        p_aug,
    )


@partial(jax.jit, static_argnames=('batch_size', 'rgb_frame_stack', 'p_aug'))
def _sample_device_bridge_batch(data, initial_locs, observation_min, observation_max,
                                rng, batch_size, rgb_frame_stack, p_aug):
    rng, idx_rng, aug_rng = jax.random.split(rng, 3)
    idxs = jax.random.randint(
        idx_rng,
        (batch_size,),
        minval=0,
        maxval=data['observations'].shape[0],
    )
    batch = _build_device_bridge_batch(
        data,
        initial_locs,
        observation_min,
        observation_max,
        idxs,
        aug_rng,
        rgb_frame_stack,
        p_aug,
    )
    return rng, batch


class DeviceBridgeDataset:
    """GPU-resident bridge dataset that stacks RGB frames per sampled batch."""

    REQUIRED_KEYS = (
        'observations',
        'next_observations',
        'actions',
        'terminals',
        'third_person_observations',
        'next_third_person_observations',
    )

    OPTIONAL_KEYS = ('rewards', 'masks')

    @classmethod
    def create_from_dataset(cls, dataset, seed, rgb_frame_stack=3, p_aug=None):
        missing_keys = [key for key in cls.REQUIRED_KEYS if key not in dataset]
        if missing_keys:
            raise KeyError(f'Missing bridge dataset keys for device cache: {missing_keys}')
        if dataset.normalized_obs_min is None or dataset.normalized_obs_max is None:
            raise ValueError('DeviceBridgeDataset requires normalize_observations() to be called first.')

        data = {}
        for key in cls.REQUIRED_KEYS:
            data[key] = jax.device_put(np.asarray(dataset[key]))
        for key in cls.OPTIONAL_KEYS:
            if key in dataset:
                data[key] = jax.device_put(np.asarray(dataset[key]))

        idxs = np.arange(dataset.size)
        next_action_idxs = np.minimum(idxs + 1, dataset.size - 1)
        data['next_actions'] = jax.device_put(np.asarray(dataset['actions'][next_action_idxs]))

        observation_min = jax.device_put(np.asarray(dataset.normalized_obs_min, dtype=np.float32))
        observation_max = jax.device_put(np.asarray(dataset.normalized_obs_max, dtype=np.float32))
        initial_locs = jax.device_put(np.asarray(dataset.initial_locs, dtype=np.int32))

        device_dataset = cls(
            data=data,
            initial_locs=initial_locs,
            observation_min=observation_min,
            observation_max=observation_max,
            rng=jax.random.PRNGKey(seed),
            rgb_frame_stack=rgb_frame_stack,
            p_aug=p_aug,
        )
        device_dataset.block_until_ready()
        return device_dataset

    def __init__(self, data, initial_locs, observation_min, observation_max,
                 rng, rgb_frame_stack, p_aug):
        self._dict = data
        self.initial_locs = initial_locs
        self.normalized_obs_min = observation_min
        self.normalized_obs_max = observation_max
        self.rng = rng
        self.rgb_frame_stack = int(rgb_frame_stack)
        self.p_aug = None if p_aug is None else float(p_aug)
        self.size = int(data['observations'].shape[0])

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def items(self):
        return self._dict.items()

    def sample(self, batch_size: int, idxs=None):
        if idxs is not None:
            self.rng, batch_rng = jax.random.split(self.rng)
            return _device_bridge_batch_from_idxs(
                self._dict,
                self.initial_locs,
                self.normalized_obs_min,
                self.normalized_obs_max,
                jnp.asarray(idxs, dtype=jnp.int32),
                batch_rng,
                self.rgb_frame_stack,
                self.p_aug,
            )

        self.rng, batch = _sample_device_bridge_batch(
            self._dict,
            self.initial_locs,
            self.normalized_obs_min,
            self.normalized_obs_max,
            self.rng,
            batch_size,
            self.rgb_frame_stack,
            self.p_aug,
        )
        return batch

    def normalize_observations(self, observations=None):
        if observations is None:
            raise ValueError('DeviceBridgeDataset cannot normalize observations in place.')
        return observations

    def nbytes(self):
        total = 0
        for value in self._dict.values():
            total += int(getattr(value, 'nbytes', 0))
        total += int(getattr(self.initial_locs, 'nbytes', 0))
        total += int(getattr(self.normalized_obs_min, 'nbytes', 0))
        total += int(getattr(self.normalized_obs_max, 'nbytes', 0))
        return total

    def block_until_ready(self):
        for leaf in jax.tree_util.tree_leaves((
            self._dict,
            self.initial_locs,
            self.normalized_obs_min,
            self.normalized_obs_max,
        )):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.obs_norm_type = 'none'  # Observation normalization type.
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.num_aug = 1  # Number of image augmentations; set outsize the class.
        self.inplace_aug = False  # Whether to replace the original image after applying augmentations.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.aug_keys = ('observations', 'next_observations')  # Image keys to augment.
        self.aux_frame_stack = None  # Frame stack count for auxiliary image keys.
        self.aux_frame_stack_keys = ()  # Current-observation auxiliary keys to stack.

        self._prestacked = False

        # observation statistics
        self.obs_mean = None
        self.obs_var = None
        self.obs_max = None
        self.obs_min = None
        self.normalized_obs_max = None
        self.normalized_obs_min = None
        self.epsilon = 1e-8  # for normalization

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    @staticmethod
    def normalize(observations, obs_mean, obs_var, obs_max, obs_min,
                  normalizer_type='none', epsilon=1e-8):
        if normalizer_type == 'normal':
            return (observations - obs_mean) / np.sqrt(
                obs_var + epsilon
            )
        elif normalizer_type == 'bounded':
            return 2 * (observations - obs_min) / (
                obs_max - obs_min
            ) - 1.0
        elif normalizer_type == 'none':
            return observations
        else:
            raise TypeError("Unsupported normalizer type: {}".format(
                normalizer_type))

    def normalize_observations(self, observations=None):
        if observations is None:
            self.obs_mean = np.mean(self['observations'], axis=0)
            self.obs_var = np.var(self['observations'], axis=0)
            self.obs_max = np.max(self['observations'], axis=0)
            self.obs_min = np.min(self['observations'], axis=0)

            self.normalized_obs_max = self.normalize(
                self.obs_max, self.obs_mean, self.obs_var,
                self.obs_max, self.obs_min,
                self.obs_norm_type, self.epsilon
            )
            self.normalized_obs_min = self.normalize(
                self.obs_min, self.obs_mean, self.obs_var,
                self.obs_max, self.obs_min,
                self.obs_norm_type, self.epsilon
            )

            assert 'observations' in self
            assert 'next_observations' in self

            observations = self['observations']

            self._dict['observations'] = self.normalize(
                self['observations'], self.obs_mean, self.obs_var,
                self.obs_max, self.obs_min,
                self.obs_norm_type, self.epsilon
            )
            self._dict['next_observations'] = self.normalize(
                self['next_observations'], self.obs_mean, self.obs_var,
                self.obs_max, self.obs_min,
                self.obs_norm_type, self.epsilon
            )

        observations = self.normalize(
            observations, self.obs_mean, self.obs_var,
            self.obs_max, self.obs_min,
            self.obs_norm_type, self.epsilon
        )

        return observations

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def _prestack_frames(self):
        """Preprocess for frame stacking -- avoid much delay in batch loading."""
        # Stack frames.
        idxs = np.arange(self.size)
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
        next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
        for i in reversed(range(self.frame_stack)):
            # Use the initial state if the index is out of bounds.
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            if i != self.frame_stack - 1:
                next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
        next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

        self._dict['observations'] = jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *obs)
        self._dict['next_observations'] = jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *next_obs)
        # don't need to prestack once we've already prestacked...
        self._prestacked = True

    def _stack_aux_frames(self, batch, idxs):
        """Stack auxiliary frame keys without changing low-dimensional observations."""
        if self.aux_frame_stack is None or not self.aux_frame_stack_keys:
            return

        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        for key in self.aux_frame_stack_keys:
            next_key = f'next_{key}'
            if key not in self or next_key not in self:
                raise KeyError(
                    f'Cannot auxiliary-frame-stack {key!r}: dataset must contain both '
                    f'{key!r} and {next_key!r}.'
                )

            obs = []  # [x[t - stack + 1], ..., x[t]]
            next_obs = []  # [x[t - stack + 2], ..., x[t], x[t + 1]]
            for i in reversed(range(self.aux_frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self[key]))
                if i != self.aux_frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self[key]))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self[next_key]))

            batch[key] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch[next_key] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        # prestack frames for faster batch sampling
        # warning: require a large amount of cpu mems
        if (self.frame_stack is not None) and (not self._prestacked):
            self._prestack_frames()
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        batch['observation_min'] = self.normalized_obs_min
        batch['observation_max'] = self.normalized_obs_max
        if (self.frame_stack is not None) and (not self._prestacked):
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        self._stack_aux_frames(batch, idxs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                aug_keys = [key for key in self.aug_keys if key in batch]
                if not aug_keys:
                    return batch
                if self.inplace_aug:
                    augment(batch, aug_keys)
                else:
                    for i in range(self.num_aug):
                        augment(batch, aug_keys, 'aug{}_'.format(i + 1))

        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

        # Update terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def add_transitions(self, transitions):
        batch_size = transitions['observations'].shape[0]
        idxs = (np.arange(batch_size) + self.pointer) % self.max_size  # (B,)

        def set_idxs(buffer, new_elements):
            buffer[idxs] = new_elements

        jax.tree_util.tree_map(set_idxs, self._dict, transitions)

        self.pointer = (self.pointer + batch_size) % self.max_size
        self.size = min(self.max_size, self.size + batch_size)

        # Update terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
            self.config['value_geom_start_offset'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
            self.config['actor_geom_start_offset'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['relabeled_masks'] = 1.0 - successes
        batch['relabeled_rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
        if self.config['relabel_reward']:
            batch['masks'] = batch['relabeled_masks']
            batch['rewards'] = batch['relabeled_rewards']

        if self.config['p_aug'] is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.config['p_aug']:
                augment(batch, ['value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, num_goals, geom_start=1):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # truncated geometric sampling.
            support_shift = geom_start - 1
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size) + support_shift  # in [0, inf) or [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            if num_goals > 1:
                middle_goal_idxs = np.round(
                    (np.minimum(idxs[:, None] + 1, final_state_idxs[:, None]) * distances + final_state_idxs[:, None] * (1 - distances))
                ).astype(int)
            else:
                middle_goal_idxs = np.round(
                    (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
                ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), middle_goal_idxs, random_goal_idxs
        )

        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def normalize_observations(self, observations=None):
        return self.dataset.normalize_observations(observations)

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)
