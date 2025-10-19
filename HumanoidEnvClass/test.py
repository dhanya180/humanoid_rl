import numpy as np
from pose_to_joint import skeleton_to_joint_angles
from humanoid_env import HumanoidWalkEnv

# Example of OpenPose BODY_25 output for one person
skeleton = np.array([
    [320, 100, 0.9],  # 0: Nose
    [320, 150, 0.95], # 1: Neck
    [280, 150, 0.9],  # 2: RShoulder
    [260, 180, 0.85], # 3: RElbow
    [250, 220, 0.8],  # 4: RWrist
    [360, 150, 0.9],  # 5: LShoulder
    [380, 180, 0.85], # 6: LElbow
    [390, 220, 0.8],  # 7: LWrist
    [320, 220, 0.95], # 8: MidHip
    [290, 250, 0.9],  # 9: RHip
    [285, 300, 0.85], # 10: RKnee
    [280, 360, 0.8],  # 11: RAnkle
    [350, 250, 0.9],  # 12: LHip
    [355, 300, 0.85], # 13: LKnee
    [360, 360, 0.8],  # 14: LAnkle
    [310, 90, 0.7],   # 15: REye
    [330, 90, 0.7],   # 16: LEye
    [300, 95, 0.6],   # 17: REar
    [340, 95, 0.6],   # 18: LEar
    [280, 380, 0.8],  # 19: LBigToe
    [360, 380, 0.8],  # 20: RBigToe
    [270, 380, 0.7],  # 21: LSmallToe
    [370, 380, 0.7],  # 22: RSmallToe
    [275, 370, 0.6],  # 23: LHeel
    [365, 370, 0.6]   # 24: RHeel
])

print("Skeleton shape:", skeleton.shape)

initial_pose = skeleton_to_joint_angles(skeleton)

env = HumanoidWalkEnv(r"assets\urdf\humanoid_25dof.urdf", gui=True)
obs, _ = env.reset(initial_pose=initial_pose)
