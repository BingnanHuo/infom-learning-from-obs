import os

import gc
import json
import random
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import DeviceBridgeDataset, GCDataset, Dataset, ReplayBuffer
from utils.evaluation import evaluate
from utils.flax_utils import (
    find_latest_training_checkpoint,
    restore_agent,
    restore_training_checkpoint,
    save_agent,
    save_training_checkpoint,
)
from utils.log_utils import CsvLogger, TensorBoardLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('wandb_mode', 'online', 'Wandb mode.')
flags.DEFINE_integer('enable_tensorboard', 0, 'Whether to log scalar metrics to TensorBoard.')
flags.DEFINE_string('tensorboard_dir', None, 'TensorBoard log directory. Defaults to save_dir/tensorboard.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Optional OGBench dataset directory.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('run_id', None, 'Stable run ID. If set, overrides timestamped experiment names.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_integer('resume_from_checkpoint', 0, 'Whether to auto-resume from the latest full training checkpoint.')
flags.DEFINE_string('resume_checkpoint_path', None, 'Checkpoint file or directory to resume from. Defaults to checkpoint_dir.')
flags.DEFINE_string('checkpoint_dir', None, 'Full training checkpoint directory. Defaults to save_dir/checkpoints.')
flags.DEFINE_integer('checkpoint_interval', 0, 'Full training checkpoint interval. Disabled when 0.')
flags.DEFINE_integer('checkpoint_keep', 3, 'Number of recent full training checkpoints to keep.')
flags.DEFINE_integer('checkpoint_at_end', 1, 'Whether to save a full training checkpoint at the end.')
flags.DEFINE_integer('checkpoint_on_signal', 1, 'Whether to checkpoint and exit on SIGUSR1/SIGTERM.')
flags.DEFINE_integer('checkpoint_signal_exit_code', 75, 'Exit code used after checkpointing on a signal.')

flags.DEFINE_integer('pretraining_steps', 1_000_000, 'Number of offline steps.')
flags.DEFINE_integer('pretraining_size', 1_000_000, 'Size of the dataset for pre-training.')
flags.DEFINE_integer('finetuning_steps', 500_000, 'Number of online steps.')
flags.DEFINE_integer('finetuning_size', 500_000, 'Size of the dataset for fine-tuning.')
flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1_500_000, 'Saving interval.')
flags.DEFINE_integer('save_best_eval', 0, 'Whether to save a checkpoint when the selected eval metric improves.')
flags.DEFINE_string('best_eval_metric', 'evaluation/episode.return', 'Evaluation metric used for best checkpointing.')
flags.DEFINE_enum('best_eval_mode', 'max', ['max', 'min'], 'Whether higher or lower best_eval_metric values are better.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_string('obs_norm_type', 'normal',
                    'Type of observation normalization. (none, normal, bounded)')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('num_aug', 1, 'Number of image augmentations.')
flags.DEFINE_integer('inplace_aug', 1, 'Whether to replace the original image after applying augmentations.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/infom.py', lock_config=False)


def _to_finite_float(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None

    if array.shape != ():
        return None

    value = float(array)
    if not np.isfinite(value):
        return None
    return value


def _is_better_eval(value, best_value, mode):
    if best_value is None:
        return True
    if mode == 'max':
        return value > best_value
    if mode == 'min':
        return value < best_value
    raise ValueError(f'Unsupported best_eval_mode: {mode}')


def _save_best_eval_if_improved(agent, eval_metrics, step, best_eval):
    metric_value = _to_finite_float(eval_metrics.get(FLAGS.best_eval_metric))
    if metric_value is None:
        print(
            f'Skipping best-eval checkpoint at step {step}: metric '
            f'{FLAGS.best_eval_metric!r} is missing, non-scalar, or non-finite.',
            flush=True,
        )
        return best_eval, None, False

    best_value = None if best_eval is None else best_eval['value']
    if not _is_better_eval(metric_value, best_value, FLAGS.best_eval_mode):
        return best_eval, metric_value, False

    save_agent(agent, FLAGS.save_dir, step)
    checkpoint_path = os.path.join(FLAGS.save_dir, f'params_{step}.pkl')
    best_eval = {
        'metric': FLAGS.best_eval_metric,
        'mode': FLAGS.best_eval_mode,
        'value': metric_value,
        'step': int(step),
        'checkpoint_path': checkpoint_path,
        'env_name': FLAGS.env_name,
        'seed': int(FLAGS.seed),
        'wandb_run_group': FLAGS.wandb_run_group,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    best_eval_path = os.path.join(FLAGS.save_dir, 'best_eval.json')
    with open(best_eval_path, 'w') as f:
        json.dump(best_eval, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'Updated best eval checkpoint metadata at {best_eval_path}', flush=True)
    return best_eval, metric_value, True


def _compute_state_distillation_stats(dataset, state_clip, epsilon):
    observations = np.asarray(dataset['observations'], dtype=np.float32)
    state_mean = np.mean(observations, axis=0)
    state_std = np.sqrt(np.var(observations, axis=0) + epsilon)
    state_latents = np.clip((observations - state_mean) / state_std, -state_clip, state_clip)
    return {
        'state_mean': state_mean.astype(np.float32),
        'state_std': state_std.astype(np.float32),
        'latent_min': np.min(state_latents, axis=0).astype(np.float32),
        'latent_max': np.max(state_latents, axis=0).astype(np.float32),
    }


def _serialize_replay_buffer(replay_buffer):
    if replay_buffer is None:
        return None

    return {
        'data': jax.tree_util.tree_map(np.array, replay_buffer._dict),
        'size': int(replay_buffer.size),
        'pointer': int(replay_buffer.pointer),
        'max_size': int(replay_buffer.max_size),
        'return_next_actions': bool(replay_buffer.return_next_actions),
    }


def _restore_replay_buffer(replay_buffer, state):
    if replay_buffer is None or state is None:
        return

    for key, value in state['data'].items():
        replay_buffer._dict[key][...] = value
    replay_buffer.size = int(state['size'])
    replay_buffer.pointer = int(state['pointer'])
    replay_buffer.max_size = int(state['max_size'])
    replay_buffer.return_next_actions = bool(state.get('return_next_actions', replay_buffer.return_next_actions))
    replay_buffer.terminal_locs = np.nonzero(replay_buffer['terminals'] > 0)[0]
    replay_buffer.initial_locs = np.concatenate([[0], replay_buffer.terminal_locs[:-1] + 1])


def _training_state_dict(
    step,
    rng,
    inferred_latent,
    best_eval,
    finetuning_replay_buffer,
    elapsed_before,
    first_time,
    reason,
):
    return {
        'step': int(step),
        'reason': reason,
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'mbpo_rng': None if rng is None else np.array(rng),
        'inferred_latent': inferred_latent,
        'best_eval': best_eval,
        'replay_buffer': _serialize_replay_buffer(finetuning_replay_buffer),
        'elapsed_time': float(elapsed_before + time.time() - first_time),
        'flags': get_flag_dict(),
    }


def _install_checkpoint_signal_handler(signal_state):
    def _handler(signum, _frame):
        signal_state['signum'] = signum
        print(f'Received signal {signum}; will checkpoint at the next safe point.', flush=True)

    signal.signal(signal.SIGUSR1, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main(_):
    # Set up logger.
    exp_name = FLAGS.run_id or get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    checkpoint_dir = FLAGS.checkpoint_dir or os.path.join(FLAGS.save_dir, 'checkpoints')
    resume_checkpoint_path = checkpoint_dir
    resume_checkpoint_is_local = True
    if FLAGS.resume_from_checkpoint:
        # Prefer this run's own restart checkpoint. If none exists yet, fall back
        # to an explicitly provided source checkpoint, e.g. a completed pretrain.
        if find_latest_training_checkpoint(checkpoint_dir) is None and FLAGS.resume_checkpoint_path:
            resume_checkpoint_path = FLAGS.resume_checkpoint_path
            resume_checkpoint_is_local = False
    resume_checkpoint_exists = (
        bool(FLAGS.resume_from_checkpoint)
        and find_latest_training_checkpoint(resume_checkpoint_path) is not None
    )
    if FLAGS.enable_wandb:
        _, trigger_sync = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='infom', group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    cross_modal_state_distilled = config['agent_name'] == 'cross_modal_state_distilled_infom'
    cross_modal_tcn = config['agent_name'] == 'cross_modal_tcn_infom'
    cross_modal_bridge_agent = cross_modal_state_distilled or cross_modal_tcn
    bridge_third_person_required = (
        cross_modal_bridge_agent
        or (config['agent_name'] == 'infom' and config.get('bridge_loss_weight', 0.0) > 0.0)
    )
    if cross_modal_bridge_agent and FLAGS.obs_norm_type != 'none':
        raise ValueError(
            f"{config['agent_name']} owns state normalization internally. "
            'Run with --obs_norm_type=none to avoid double-normalizing evaluation states.'
        )

    # Make environment and datasets.
    env_frame_stack = None if cross_modal_bridge_agent else FLAGS.frame_stack
    _, _, pretraining_train_dataset, pretraining_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=env_frame_stack, max_size=FLAGS.pretraining_size, reward_free=True,
        dataset_dir=FLAGS.dataset_dir, include_third_person=bridge_third_person_required)
    needs_finetuning_data = FLAGS.finetuning_steps > 0
    if needs_finetuning_data:
        _, eval_env, finetuning_train_dataset, finetuning_val_dataset = make_env_and_datasets(
            FLAGS.env_name, frame_stack=env_frame_stack, max_size=FLAGS.finetuning_size, reward_free=False,
            dataset_dir=FLAGS.dataset_dir, include_third_person=bridge_third_person_required)
    else:
        eval_env = None
        finetuning_train_dataset = None
        finetuning_val_dataset = None

    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    pretraining_train_dataset = Dataset.create(**pretraining_train_dataset)
    if finetuning_train_dataset is not None:
        finetuning_train_dataset = Dataset.create(**finetuning_train_dataset)
    cross_modal_create_kwargs = {}
    if cross_modal_bridge_agent:
        cross_modal_create_kwargs = _compute_state_distillation_stats(
            pretraining_train_dataset,
            state_clip=config['state_clip'],
            epsilon=config['state_norm_eps'],
        )
        if cross_modal_tcn:
            shared_latent_dim = config['shared_latent_dim']
            cross_modal_create_kwargs['latent_min'] = -np.ones(shared_latent_dim, dtype=np.float32)
            cross_modal_create_kwargs['latent_max'] = np.ones(shared_latent_dim, dtype=np.float32)
    finetuning_replay_buffer = None
    if config['agent_name'] == 'mbpo_rebrac':
        # Create a separate replay buffer so that we can sample from both the training dataset and imaginary rollouts.
        if finetuning_train_dataset is None:
            raise ValueError('mbpo_rebrac requires finetuning data. Run with --finetuning_steps > 0.')
        example_transition = {k: v[0] for k, v in finetuning_train_dataset.items()}
        finetuning_replay_buffer = ReplayBuffer.create(example_transition, size=100)
        finetuning_replay_buffer.return_next_actions = True
    # Set p_aug, frame_stack, and return_next_actions.
    for dataset in [pretraining_train_dataset, pretraining_val_dataset,
                    finetuning_train_dataset, finetuning_val_dataset]:
        if dataset is not None:
            dataset.obs_norm_type = 'none' if cross_modal_bridge_agent else FLAGS.obs_norm_type
            dataset.p_aug = (
                FLAGS.p_aug
                if (FLAGS.p_aug is not None or not cross_modal_bridge_agent)
                else config['rgb_p_aug']
            )
            dataset.num_aug = FLAGS.num_aug
            dataset.inplace_aug = FLAGS.inplace_aug
            dataset.frame_stack = None if cross_modal_bridge_agent else FLAGS.frame_stack
            if cross_modal_bridge_agent:
                dataset.aux_frame_stack = config['rgb_frame_stack']
                dataset.aux_frame_stack_keys = ('third_person_observations',)
                dataset.aug_keys = ('third_person_observations', 'next_third_person_observations')
            if config['agent_name'] in ['infom', 'rebrac', 'dino_rebrac', 'mbpo_rebrac',
                                        'td_infonce', 'fb_repr_fom', 'hilp_fom',
                                        'cross_modal_state_distilled_infom',
                                        'cross_modal_tcn_infom']:
                dataset.return_next_actions = True
            dataset.normalize_observations()
    if config['agent_name'] in ['crl_infonce', 'td_infonce', 'hilp']:
        config['p_aug'] = FLAGS.p_aug
        config['frame_stack'] = FLAGS.frame_stack
        pretraining_train_dataset = GCDataset(pretraining_train_dataset, config)
        finetuning_train_dataset = GCDataset(finetuning_train_dataset, config)
        if pretraining_val_dataset is not None:
            pretraining_val_dataset = GCDataset(pretraining_val_dataset, config)
            finetuning_val_dataset = GCDataset(finetuning_val_dataset, config)
    finetuning_eval_dataset = finetuning_train_dataset

    # Create agent.
    example_batch = pretraining_train_dataset.sample(1)
    create_kwargs = {}
    if config['agent_name'] == 'infom' and config.get('bridge_loss_weight', 0.0) > 0.0:
        if 'third_person_observations' not in example_batch:
            raise ValueError(
                'InFOM bridge loss requires a bridge dataset with third_person_observations. '
                'Use an env_name starting with "bridge-".'
            )
        create_kwargs['ex_third_person_observations'] = example_batch['third_person_observations']
    elif cross_modal_bridge_agent:
        if 'third_person_observations' not in example_batch:
            raise ValueError(
                f"{config['agent_name']} requires a bridge dataset with third_person_observations. "
                'Use an env_name starting with "bridge-".'
            )
        create_kwargs.update(cross_modal_create_kwargs)
        create_kwargs['ex_third_person_observations'] = example_batch['third_person_observations']

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        **create_kwargs,
    )

    best_eval = None
    inferred_latent = None  # Only for HILP and FB.
    rng = jax.random.PRNGKey(FLAGS.seed)  # Only for MBPO
    start_step = 1
    elapsed_before = 0.0
    restored_checkpoint_path = None

    # Restore agent or full training state.
    if resume_checkpoint_exists and FLAGS.restore_path is not None:
        raise ValueError('Use either restore_path/restore_epoch or resume_from_checkpoint, not both.')
    if resume_checkpoint_exists:
        agent, checkpoint, restored_checkpoint_path = restore_training_checkpoint(agent, resume_checkpoint_path)
        training_state = checkpoint['training_state']
        start_step = int(checkpoint['step']) + 1
        if resume_checkpoint_is_local:
            random.setstate(training_state['python_random_state'])
            np.random.set_state(training_state['numpy_random_state'])
            if training_state.get('mbpo_rng') is not None:
                rng = jnp.asarray(training_state['mbpo_rng'], dtype=jnp.uint32)
            inferred_latent = training_state.get('inferred_latent')
            best_eval = training_state.get('best_eval')
            elapsed_before = float(training_state.get('elapsed_time', 0.0))
            _restore_replay_buffer(finetuning_replay_buffer, training_state.get('replay_buffer'))
    elif FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    use_device_bridge_cache = (
        cross_modal_bridge_agent
        and bool(config.get('device_bridge_cache', False))
    )
    pretraining_device_cached = False
    finetuning_device_cached = False

    def cache_bridge_dataset(dataset, phase, seed_offset):
        print(f'Creating {phase} DeviceBridgeDataset cache...', flush=True)
        device_dataset = DeviceBridgeDataset.create_from_dataset(
            dataset,
            seed=FLAGS.seed + seed_offset,
            rgb_frame_stack=config['rgb_frame_stack'],
            p_aug=(
                FLAGS.p_aug
                if FLAGS.p_aug is not None
                else config['rgb_p_aug']
            ),
        )
        print(
            f'{phase} DeviceBridgeDataset cache ready: '
            f'{device_dataset.size} transitions, '
            f'{device_dataset.nbytes() / (1024 ** 3):.2f} GiB device arrays.',
            flush=True,
        )
        return device_dataset

    if use_device_bridge_cache:
        if start_step <= FLAGS.pretraining_steps:
            pretraining_train_dataset = cache_bridge_dataset(
                pretraining_train_dataset,
                phase='pretraining',
                seed_offset=17,
            )
            pretraining_device_cached = True
        else:
            pretraining_train_dataset = None
            finetuning_train_dataset = cache_bridge_dataset(
                finetuning_train_dataset,
                phase='finetuning',
                seed_offset=23,
            )
            finetuning_device_cached = True

    # Train agent.
    pretraining_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_train.csv'), append=resume_checkpoint_exists)
    pretraining_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_eval.csv'), append=resume_checkpoint_exists)
    finetuning_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_train.csv'), append=resume_checkpoint_exists)
    finetuning_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_eval.csv'), append=resume_checkpoint_exists)
    tensorboard_logger = None
    if FLAGS.enable_tensorboard:
        tensorboard_dir = FLAGS.tensorboard_dir or os.path.join(FLAGS.save_dir, 'tensorboard')
        tensorboard_logger = TensorBoardLogger(tensorboard_dir)
    first_time = time.time()
    last_time = time.time()
    total_steps = FLAGS.pretraining_steps + FLAGS.finetuning_steps
    if restored_checkpoint_path is not None:
        print(
            f'Resuming {FLAGS.env_name} seed {FLAGS.seed} from step {start_step} '
            f'using {restored_checkpoint_path}',
            flush=True,
        )

    signal_state = {'signum': None}
    if FLAGS.checkpoint_on_signal:
        _install_checkpoint_signal_handler(signal_state)

    def save_full_checkpoint(step, reason):
        if step <= 0:
            return None
        state = _training_state_dict(
            step=step,
            rng=rng,
            inferred_latent=inferred_latent,
            best_eval=best_eval,
            finetuning_replay_buffer=finetuning_replay_buffer,
            elapsed_before=elapsed_before,
            first_time=first_time,
            reason=reason,
        )
        return save_training_checkpoint(agent, checkpoint_dir, step, state, keep=FLAGS.checkpoint_keep)

    for i in tqdm.tqdm(range(start_step, total_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.pretraining_steps:
            # Offline pre-training.
            batch = pretraining_train_dataset.sample(config['batch_size'])
            train_logger = pretraining_train_logger
            eval_logger = pretraining_eval_logger

            agent, update_info = agent.pretrain(batch)
        else:
            if i == (FLAGS.pretraining_steps + 1):
                if use_device_bridge_cache and not finetuning_device_cached:
                    if pretraining_device_cached:
                        print('Releasing pretraining DeviceBridgeDataset cache before fine-tuning.', flush=True)
                        pretraining_train_dataset = None
                        pretraining_device_cached = False
                        gc.collect()
                    finetuning_train_dataset = cache_bridge_dataset(
                        finetuning_train_dataset,
                        phase='finetuning',
                        seed_offset=23,
                    )
                    finetuning_device_cached = True

                if config['agent_name'] in [
                    'infom',
                    'dino_rebrac',
                    'td_infonce',
                    'hilp',
                    'cross_modal_state_distilled_infom',
                    'cross_modal_tcn_infom',
                ]:
                    agent.target_reset()

                # Infer the latent vector.
                if config['agent_name'] in ['hilp', 'fb_repr']:
                    num_samples = 0
                    inference_batch = defaultdict(list)
                    while num_samples < config['num_latent_inference_samples']:
                        batch = finetuning_train_dataset.sample(config['batch_size'])
                        for k, v in batch.items():
                            inference_batch[k].append(v)
                        num_samples += config['batch_size']
                    for k, v in inference_batch.items():
                        if k not in ['observation_min', 'observation_max']:
                            inference_batch[k] = np.concatenate(v, axis=0)[:config['num_latent_inference_samples']]

                    inferred_latent = agent.infer_latent(inference_batch)
                    inferred_latent = np.array(inferred_latent)

            # Offline fine-tuning.
            if (config['agent_name'] == 'mbpo_rebrac') and (finetuning_replay_buffer.size > config['batch_size']):
                # Half-and-half sampling from the training dataset and the replay buffer.
                batch = finetuning_train_dataset.sample(config['batch_size'])
                replay_batch = finetuning_replay_buffer.sample(config['batch_size'])
                for k, v in replay_batch.items():
                    batch[f'model_{k}'] = v
            else:
                # batch = pretraining_train_dataset.sample(config['batch_size'])
                batch = finetuning_train_dataset.sample(config['batch_size'])
            train_logger = finetuning_train_logger
            eval_logger = finetuning_eval_logger

            if config['agent_name'] in ['hilp', 'fb_repr']:
                batch['latents'] = np.tile(inferred_latent, (batch['observations'].shape[0], 1))

            agent, update_info = agent.finetune(batch, full_update=(i % config['actor_freq'] == 0))

        # MBPO imaginary rollouts
        if config['agent_name'] in ['mbpo_rebrac'] and (i > FLAGS.pretraining_steps):
            batch = finetuning_train_dataset.sample(config['num_model_rollouts'])
            observations = batch['observations']
            for _ in range(config['num_model_rollout_steps']):
                rng, actor_rng = jax.random.split(rng)

                actions = agent.sample_actions(observations=observations, temperature=1, seed=actor_rng)
                rewards = agent.predict_rewards(observations=observations, actions=actions)
                next_observations = agent.predict_next_observations(observations=observations, actions=actions)

                finetuning_replay_buffer.add_transitions(
                    dict(
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        terminals=np.zeros_like(rewards),
                        masks=np.ones_like(rewards),
                        next_observations=next_observations,
                    )
                )

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if i <= FLAGS.pretraining_steps:
                val_dataset = pretraining_val_dataset
                loss_fn = agent.pretraining_loss
            else:
                val_dataset = finetuning_val_dataset
                loss_fn = agent.finetuning_loss
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])

                if config['agent_name'] in ['hilp', 'fb_repr'] and (inferred_latent is not None):
                    val_batch['latents'] = np.tile(inferred_latent, (val_batch['observations'].shape[0], 1))

                _, val_info = loss_fn(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = elapsed_before + time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            train_logger.log(train_metrics, step=i)
            if tensorboard_logger is not None:
                tensorboard_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if (FLAGS.eval_interval != 0 and (i > FLAGS.pretraining_steps)
            and (i == (FLAGS.pretraining_steps + 1) or i % FLAGS.eval_interval == 0)):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                dataset=finetuning_eval_dataset,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                inferred_latent=inferred_latent,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.save_best_eval:
                best_eval, best_eval_metric_value, is_best_eval = _save_best_eval_if_improved(
                    agent, eval_metrics, i, best_eval)
                if best_eval_metric_value is not None:
                    eval_metrics['best_eval/current_value'] = best_eval_metric_value
                if best_eval is not None:
                    eval_metrics['best_eval/best_value'] = best_eval['value']
                    eval_metrics['best_eval/best_step'] = best_eval['step']
                eval_metrics['best_eval/is_best'] = float(is_best_eval)

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()
            eval_logger.log(eval_metrics, step=i)
            if tensorboard_logger is not None:
                tensorboard_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
        if FLAGS.checkpoint_interval > 0 and i % FLAGS.checkpoint_interval == 0:
            save_full_checkpoint(i, reason='interval')
        if signal_state['signum'] is not None:
            save_full_checkpoint(i, reason=f'signal_{signal_state["signum"]}')
            pretraining_train_logger.close()
            pretraining_eval_logger.close()
            finetuning_train_logger.close()
            finetuning_eval_logger.close()
            if tensorboard_logger is not None:
                tensorboard_logger.close()
            raise SystemExit(FLAGS.checkpoint_signal_exit_code)

    if FLAGS.checkpoint_at_end:
        save_full_checkpoint(total_steps, reason='end')

    pretraining_train_logger.close()
    pretraining_eval_logger.close()
    finetuning_train_logger.close()
    finetuning_eval_logger.close()
    if tensorboard_logger is not None:
        tensorboard_logger.close()


if __name__ == '__main__':
    app.run(main)
