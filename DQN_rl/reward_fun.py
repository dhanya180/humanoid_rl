import numpy as np

def calculate_reward(observations, info, torques, isDone, w_vel=1.0, w_live=0.1, w_energy=0.01):
    # computing the forward velocity along X-axis

    torso_lin_vel = info.get('torso_lin_vel', np.array([0.0,0.0,0.0]))
    r_vel = torso_lin_vel[0]

    # bonus if the robo is still alive that is the torso height is above threshold
    base_pos = info.get('torso_pos', np.array([0.0,0.0,1.0]))
    r_live = 1.0 if base_pos[2] > 0.5 else 0.0

    # energy penalty
    r_energy = np.sum(np.square(torques))
    reward = w_vel * r_vel + w_live * r_live - w_energy * r_energy

    if isDone:
        reward -= 5.0

    return reward

