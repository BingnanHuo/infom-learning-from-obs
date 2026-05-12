import os

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import GCDataset, Dataset, ReplayBuffer
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
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
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

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


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.enable_wandb:
        _, trigger_sync = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='infom', group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    _, _, pretraining_train_dataset, pretraining_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=FLAGS.pretraining_size, reward_free=True,
        dataset_dir=FLAGS.dataset_dir)
    _, eval_env, finetuning_train_dataset, finetuning_val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=FLAGS.finetuning_size, reward_free=False,
        dataset_dir=FLAGS.dataset_dir)

    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    pretraining_train_dataset = Dataset.create(**pretraining_train_dataset)
    finetuning_train_dataset = Dataset.create(**finetuning_train_dataset)
    if config['agent_name'] == 'mbpo_rebrac':
        # Create a separate replay buffer so that we can sample from both the training dataset and imaginary rollouts.
        example_transition = {k: v[0] for k, v in finetuning_train_dataset.items()}
        finetuning_replay_buffer = ReplayBuffer.create(example_transition, size=100)
        finetuning_replay_buffer.return_next_actions = True
    # Set p_aug, frame_stack, and return_next_actions.
    for dataset in [pretraining_train_dataset, pretraining_val_dataset,
                    finetuning_train_dataset, finetuning_val_dataset]:
        if dataset is not None:
            dataset.obs_norm_type = FLAGS.obs_norm_type
            dataset.p_aug = FLAGS.p_aug
            dataset.num_aug = FLAGS.num_aug
            dataset.inplace_aug = FLAGS.inplace_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ['infom', 'rebrac', 'dino_rebrac', 'mbpo_rebrac',
                                        'td_infonce', 'fb_repr_fom', 'hilp_fom']:
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

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        **create_kwargs,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    pretraining_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_train.csv'))
    pretraining_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_eval.csv'))
    finetuning_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_train.csv'))
    finetuning_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_eval.csv'))
    tensorboard_logger = None
    if FLAGS.enable_tensorboard:
        tensorboard_dir = FLAGS.tensorboard_dir or os.path.join(FLAGS.save_dir, 'tensorboard')
        tensorboard_logger = TensorBoardLogger(tensorboard_dir)
    first_time = time.time()
    last_time = time.time()
    best_eval = None

    inferred_latent = None  # Only for HILP and FB.
    rng = jax.random.PRNGKey(FLAGS.seed)  # Only for MBPO
    for i in tqdm.tqdm(range(1, FLAGS.pretraining_steps + FLAGS.finetuning_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.pretraining_steps:
            # Offline pre-training.
            batch = pretraining_train_dataset.sample(config['batch_size'])
            train_logger = pretraining_train_logger
            eval_logger = pretraining_eval_logger

            agent, update_info = agent.pretrain(batch)
        else:
            if i == (FLAGS.pretraining_steps + 1):
                if config['agent_name'] in ['infom', 'dino_rebrac', 'td_infonce', 'hilp']:
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
            train_metrics['time/total_time'] = time.time() - first_time
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
                dataset=finetuning_train_dataset,
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

    pretraining_train_logger.close()
    pretraining_eval_logger.close()
    finetuning_train_logger.close()
    finetuning_eval_logger.close()
    if tensorboard_logger is not None:
        tensorboard_logger.close()


if __name__ == '__main__':
    app.run(main)
