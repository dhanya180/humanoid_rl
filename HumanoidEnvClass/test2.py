import numpy as np
import time
from humanoid_env import HumanoidWalkEnv  # your env class file
from pose_to_joint import skeleton_to_joint_angles  # your conversion function

# 1️⃣ Example skeleton (like from OpenPose)
skeleton = np.array([
    [320, 100, 0.9], [320, 150, 0.95], [280, 150, 0.9], [260, 180, 0.85], [250, 220, 0.8],
    [360, 150, 0.9], [380, 180, 0.85], [390, 220, 0.8], [320, 220, 0.95], [290, 250, 0.9],
    [285, 300, 0.85], [280, 360, 0.8], [350, 250, 0.9], [355, 300, 0.85], [360, 360, 0.8],
    [310, 90, 0.7], [330, 90, 0.7], [300, 95, 0.6], [340, 95, 0.6], [280, 380, 0.8],
    [360, 380, 0.8], [270, 380, 0.7], [370, 380, 0.7], [275, 370, 0.6], [365, 370, 0.6]
])

# 2️⃣ Convert skeleton to initial joint angles
θ_init = skeleton_to_joint_angles(skeleton)
print("Initial pose vector (θ_init):", θ_init)

# 3️⃣ Initialize your environment
env = HumanoidWalkEnv(gui=True)  # pass gui=True to see it in PyBullet

# 4️⃣ Reset simulation to that pose
obs = env.reset(initial_pose=θ_init)

# 5️⃣ Run a few random steps
for step in range(100000):
    action = env.action_space.sample()   # random action for testing
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(1/60)
    if done or truncated:
        print("Episode ended at step:", step)
        break

env.close()
