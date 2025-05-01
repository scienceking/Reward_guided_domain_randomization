"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch

from arguments import get_args
from ppo2 import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
from random_pendulum import RandomizedPendulumEnv
import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.registration import register
register(
    id='RandomizedPendulumEnv-v0',
    entry_point=RandomizedPendulumEnv,
)

def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)


	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# # Tries to load in an existing actor/critic model to continue training on
	# if actor_model != '' and critic_model != '':
	# 	print(f"Loading in {actor_model} and {critic_model}...", flush=True)
	# 	model.actor.load_state_dict(torch.load(actor_model))
	# 	model.critic.load_state_dict(torch.load(critic_model))
	# 	print(f"Successfully loaded.", flush=True)
	# elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
	# 	print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
	# 	sys.exit(0)
	# else:
	# 	print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	reward_set =model.learn(total_timesteps= 2000000*1)

	return reward_set

def test(env, actor_model, episodes=1000, max_steps_per_episode=200):
    """
    Tests the trained PPO policy over a number of episodes, capped at max_steps_per_episode.

    Parameters:
        env - the Gymnasium environment to test in
        actor_model - path to the saved actor model
        episodes - number of episodes to run
        max_steps_per_episode - cap per episode steps (default: 50)

    Returns:
        avg_reward - average total reward over episodes
    """
    import numpy as np
    import torch
    import sys

    print(f"Testing {actor_model} over {episodes} episodes", flush=True)

    if actor_model == '':
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))
    policy.eval()

    total_reward = 0.0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            obs = np.asarray(obs)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_tensor).squeeze(0).numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_count += 1

        print(f"Episode {ep + 1} reward: {ep_reward}")
        total_reward += ep_reward

    avg_reward = total_reward / episodes
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward}")
    return avg_reward


def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 2000,
				'max_timesteps_per_episode': 200, 
				'gamma': 0.9,
				'n_updates_per_iteration': 10,
				'lr': 8e-5,
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
	#env = gym.make('Pendulum-v1', render_mode='human' if args.mode == 'test' else 'rgb_array')
	env = gym.make('RandomizedPendulumEnv-v0', render_mode='rgb_array' if args.mode == 'test' else 'rgb_array')

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		reward_set = train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		reward_set = test(env=env, actor_model=args.actor_model, episodes=1000)

	return reward_set

if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    reward_set, T_time = main(args)

    import csv

    # 保存 reward_set
    reward_file = "reward7.csv"
    with open(reward_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss'])
        for epoch, loss_val in enumerate(reward_set, start=1):
            writer.writerow([epoch, loss_val])

    # 保存每次 reset 的耗时
    T_time_file = "T_time7.csv"
    with open(T_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Reset Index', 'Time (s)'])
        for i, t in enumerate(T_time, start=1):
            writer.writerow([i, t])



