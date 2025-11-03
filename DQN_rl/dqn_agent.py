import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque   # to implement circular buffer for storing experiences

n_joints = 20   # defining the number of joints as per the number of joints being used in the Humanoid Environment class
n_bins = 5  # defining the number of bins according the number specified in the document

# defining the Q network , a fully conneted neural network(that generalizes any function and hence the Q learning function as well) with 3 layers and the internal layers having 512 neurons.
# We are using ReLu as a non-linear activation function.

class Q_Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fully_connected1 = nn.Linear(state_dim,512)  # 512 neurons
        self.fully_connected2 = nn.Linear(512,512)
        self.fully_connected3 = nn.Linear(512,action_dim)

    def forward(self,x):
        x = torch.relu(self.fully_connected1(x))
        x = torch.relu(self.fully_connected2(x))
        return self.fully_connected3(x)

class DQNAgent:
    def __init__(self,state_dim,action_dim,lr=1e-3,gamma=0.99,epsilon=1.0,epsilon_decay=0.995,buffer_size=100000,batch_size=64):
        self.state_dim = state_dim
        self.action_dim = n_joints * n_bins
        self.num_joints = n_joints
        self.num_bins = n_bins
        self.Q_net = Q_Network(state_dim, action_dim)
        self.target_net = Q_Network(state_dim, action_dim)
        self.target_net.load_state_dict(self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=lr)
        # choosing Mean Squared error loss
        self.loss_fn = nn.MSELoss()

        # the discounted value that will be used in computing the next target values using the function (Q_net)
        self.gamma = gamma

        # hyper parameters for tuning between exploration and exploitation
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        # intialising the circular buffer to ensure that we are using the past experiences for training, which lets the user learn and get trained as well.
        # we will be storing all the previous states, actions performed, reward obtained for that action, next state we transitioned to and a Boolean value that indicates if the model reached terminal condition or not
        self.replay_buffer = deque(maxlen=buffer_size)

        self.batch_size = batch_size

        # compute the number of steps taken for updating the target Q-network that will be providing the values for enabling the Q-network to learn from the past experiences
        self.steps = 0
        self.update_target_steps = 1000     # synchronise the target Q network periodically with this 1000 steps being the period

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_indices = np.random.randint(0, self.num_bins, size=self.num_joints)
            torque_bins = np.array([-1.0, -0.5, 0, 0.5, 1.0])
            # return torque_bins[action_indices]
            return action_indices

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)    # convert the state vector into a tensor to pass it as input to the Q network
        with torch.no_grad():
            q_table_val = self.Q_net(state_tensor)  # shape will be [1,125]
        # return q_table_val.argmax().item()
        q_table_val = q_table_val.view(self.num_joints, self.num_bins)      # shape will be [25,5]

        # pick up the arg max for each joint
        actions_idx = q_table_val.argmax(dim=1).cpu().numpy()   # an array of length 25 that will be the chosen actions that are having max value

        # mapping bin indices to actual torque values
        # torque_bins = np.array([-1.0,-0.5,0,0.5,1.0])
        # action_torques = torque_bins[actions_idx]

        # return action_torques
        return actions_idx

    def store_transition_cir_buf(self, state, action, reward, next_state, isDone):
        self.replay_buffer.append((state,action,reward,next_state,isDone))

    def update_Q_net(self):
        # wait until we accumulate atleast batch size number of experiences
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)  # select a random sample form the buffer size
        states, actions, rewards, next_states, isDones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = np.array(actions)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        isDones = torch.tensor(isDones, dtype=torch.float32).unsqueeze(1)

        q_table_values = self.Q_net(states).view(-1, self.num_joints, self.num_bins)
        actions = actions.unsqueeze(2)     # shape of the batch is [25,1]
        q_selected_val = q_table_values.gather(2,actions).squeeze(2)    # shape will be [batch,25]

        with torch.no_grad():
            # q_next_val = self.target_net(next_states).max(1)[0].unsqueeze(1)
            next_q_vals = self.target_net(next_states).view(-1, self.num_joints, self.num_bins)
            max_next_q_vals = next_q_vals.max(dim=2)[0]


        rewards = rewards.expand(-1, self.num_joints)
        isDones = isDones.expand(-1, self.num_joints)

        q_target = rewards + self.gamma * max_next_q_vals * (1-isDones)

        loss = self.loss_fn(q_selected_val, q_target)
        self.optimizer.zero_grad()  # clears the gradients that are computed before
        loss.backward()
        self.optimizer.step()

        # perform epsilon decay to encourage exploitation under low epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update the target Q network periodically
        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.Q_net.state_dict())