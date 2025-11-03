import torch
import os
import numpy as np

from HumanoidEnvClass.humanoid_env import HumanoidWalkEnv
from pose_estimation.extractor import PoseEstimatorModule1
from DQN_rl.dqn_agent import DQNAgent
from DQN_rl.reward_fun import calculate_reward


# defining the hyperparameters

BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 100000
LR = 1e-3
NUM_JOINTS = 20
NUM_BINS = 5
NUM_EPISODES = 500  # total number of episodes
MAX_STEPS = 1000    # number steps that are being run per episode

# Initialising the Pose extractor
# Initially we load the image and then run the processing on that image
# Then we get the vector of identified initial pose (if we are not able to get any pose then fall back to zeros)

pose_est = PoseEstimatorModule1()
img_path = "/content/test_5.jpg"
_, initial_pose_vector = pose_est.process_image(img_path)
# pose_est.close()

if initial_pose_vector is None:
    # if we are not able to extract any of the skeleton then fallback to zeros
    initial_pose_vector = np.zeros(NUM_JOINTS)

# Initialising the Humanoid Walking Environment

environment = HumanoidWalkEnv(gui=False)
state_space_dimensions = environment.observation_space.shape[0]
action_space_dimensions = NUM_JOINTS * NUM_BINS

# Initialising the DQN agent that takes care of learning

dqn_agent = DQNAgent(
    state_dim=state_space_dimensions,
    action_dim=action_space_dimensions,
    lr=LR,
    gamma=GAMMA,
    epsilon=EPSILON_START,
    epsilon_decay=EPSILON_DECAY,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE
)

# Before proceeding with the training loop, run the extractor module on different images and then store each theta vector in a list (a list of lists) 

# Training loop

for episode in range(NUM_EPISODES):
    state, info = environment.reset(initial_pose=initial_pose_vector)
    episode_reward = 0

    for step in range(MAX_STEPS):
        # Select an action according to the current state of the environment
        action_indices = dqn_agent.select_action(state)

        torque_bins = np.array([-1.0, -0.5, 0, 0.5, 1.0])
        action = torque_bins[action_indices]

        # get the step that will be taken when the robo is in the environment
        next_state, _, done, truncated, next_info = environment.step(action)

        # compute the reward using the module reward_fun
        reward = calculate_reward(next_state, next_info, action, done)

        # store the transition in the defined circular buffer
        dqn_agent.store_transition_cir_buf(state,action_indices,reward,next_state,done)

        # update the Q network

        dqn_agent.update_Q_net()

        state = next_state
        info = next_info
        episode_reward += reward

        if done or truncated:
            break

        print(f"Episode {episode+1}/{NUM_EPISODES} | Total Reward : {episode_reward:.2f} | Epsilon : {dqn_agent.epsilon:.3f}")

        # Save model after every 50 episodes
        if (episode + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(dqn_agent.Q_net.state_dict(), f"models/dqn_qnet_ep{episode+1}.pth")
            print(f"Saved model at episode : {episode+1}")


# environment.close()