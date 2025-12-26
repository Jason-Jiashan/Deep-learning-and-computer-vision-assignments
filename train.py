# train.py
# Training script for the reinforcement learning agent (DQN) controlling the traffic signal.

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from traffic_models.agent import DQNAgent
from traffic_utils.env import TrafficEnv
from traffic_utils.replay_buffer import ReplayBuffer

def train_dqn(episodes=100, max_steps=100):
    # Set up environment and agent
    env = TrafficEnv(max_steps=max_steps)  # create environment with random traffic pattern
    state_dim = 3    # [queue_NS, queue_EW, phase]
    action_dim = 2   # [keep phase, switch phase]
    agent = DQNAgent(state_dim, action_dim)
    target_agent = DQNAgent(state_dim, action_dim)
    target_agent.load_state_dict(agent.state_dict())  # initialize target network
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    # Experience replay buffer (with Prioritized ER enabled)
    buffer = ReplayBuffer(capacity=10000, prioritized=True, alpha=0.6)
    # Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995  # decay per episode
    batch_size = 64
    target_update_interval = 100  # steps between target network updates
    # Setup TensorBoard logger
    writer = SummaryWriter(log_dir="runs/traffic_rl")
    print("Starting DQN training for traffic signal control...")
    global_step = 0

    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                # Choose best action from Q-network
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_vals = agent(state_tensor)
                action = int(torch.argmax(q_vals, dim=1).item())
            # Apply action in the environment
            next_state, reward, done = env.step(action)
            ep_reward += reward
            # Store transition in replay buffer
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            # Train the DQN agent using a batch from replay buffer
            if len(buffer.buffer) >= batch_size:
                batch, indices = buffer.sample(batch_size)
                # Convert batch to tensors
                states = torch.FloatTensor([b[0] for b in batch])
                actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)  # shape (batch,1)
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([b[3] for b in batch])
                dones = torch.FloatTensor([b[4] for b in batch])
                # Compute current Q values and target Q values
                q_values = agent(states).gather(1, actions).squeeze(1)  # Q(s,a) for taken actions
                with torch.no_grad():
                    # Q_target for next state
                    max_next_q = target_agent(next_states).max(dim=1)[0]  # max Q across actions
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                # Compute loss (MSE)
                loss = torch.nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Update priorities in buffer (PER) using absolute TD errors
                td_errors = (q_values - target_q).detach().cpu().numpy()
                buffer.update_priorities(indices, td_errors)
                # Periodically update target network
                if global_step % target_update_interval == 0:
                    target_agent.load_state_dict(agent.state_dict())
            global_step += 1
            if done:
                break
        # Decay exploration rate
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        # Log episode reward
        writer.add_scalar("EpisodeReward", ep_reward, ep)
        print(f"Episode {ep}/{episodes} - Reward: {ep_reward:.2f}")
    # Training finished
    writer.close()
    # Save the trained DQN model
    os.makedirs("runs", exist_ok=True)
    torch.save(agent.state_dict(), "runs/dqn_agent.pth")
    print("Training completed. Model saved to runs/dqn_agent.pth")

# If run as script, parse arguments for episodes
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Train DQN agent for traffic signal control")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()
    train_dqn(episodes=args.episodes, max_steps=args.steps)
