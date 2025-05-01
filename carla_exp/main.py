from register_env import register_custom_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray
import os

# Initialize Ray
ray.init()

# Register the environment
register_custom_env()

# Configure the PPO algorithm
config = (
    PPOConfig()
    .environment("CustomEnv")  # Name of the registered environment
    .framework("torch")        # Use PyTorch
    .rollouts(num_rollout_workers=1)  # Single worker for rollouts
    .resources(num_gpus=1)  # Use GPU if available
    .training(
        train_batch_size = 2000,  # Total batch size for training
        sgd_minibatch_size=128,  # Minibatch size for SGD
        num_sgd_iter=20,  # Number of SGD iterations per batch
        lr=1e-5,  # Learning rate (1e-5 is smaller than 1e-4)
        gamma=0.99,  # Discount factor
        lambda_=0.95,  # Lambda for GAE
        clip_param=0.2,  # PPO clipping parameter
        grad_clip=0.5 # Gradient clipping value
    )
    .training(model={
        "fcnet_hiddens": [256, 256, 256],  # Three fully connected layers
        "fcnet_activation": "tanh",        # Activation function for all layers
    })
)

# Create the log directory
log_dir = "logs/ppo_custom_env"
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
checkpoint_path = "ppo_model_checkpoint/ppo_model1_85"
# Build the trainer
trainer = config.build()
# trainer.restore(checkpoint_path)

# Training loop
for i in range(3000):
    result = trainer.train()
    print(f"Iteration {i}: reward_mean = {result['episode_reward_mean']}")

    # Save training logs
    with open(f"{log_dir}/training_log.txt", "a") as log_file:
        log_file.write(f"Iteration {i}: {result}\n")

    # Save the model every 5 iterations
    if i % 5 == 0:
        # Create a directory for saving checkpoints
        checkpoint_dir = "ppo_model_checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set the checkpoint path, including iteration number
        checkpoint_path = os.path.join(checkpoint_dir, f"ppo_model1_{85+i}")

        # Save the model
        trainer.save(checkpoint_path)

# Shutdown Ray
ray.shutdown()

