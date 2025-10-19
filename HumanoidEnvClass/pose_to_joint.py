import numpy as np

def skeleton_to_joint_angles(skeleton):
    """
    Convert a 25-keypoint skeleton to a 25-joint initial pose vector for the humanoid.
    
    Parameters:
        skeleton: np.array of shape [25, 3] -> (x, y, confidence)
    
    Returns:
        initial_pose: np.array of shape [25], joint angles in radians
    """
    # Initialize with zeros
    initial_pose = np.zeros(25)

    # Define keypoint indices (OpenPose BODY_25)
    # Example mapping (can adjust based on your URDF joint order):
    # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist
    # 5: LShoulder, 6: LElbow, 7: LWrist
    # 8: MidHip, 9: RHip, 10: RKnee, 11: RAnkle
    # 12: LHip, 13: LKnee, 14: LAnkle
    # 15-24: Eyes, Ears, Feet tips etc. (optional)

    # Torso / spine angles
    neck = skeleton[1]
    mid_hip = skeleton[8]
    spine_vec = neck[:2] - mid_hip[:2]
    spine_angle = np.arctan2(spine_vec[1], spine_vec[0])
    initial_pose[0] = spine_angle   # spine1_joint
    initial_pose[1] = 0.0           # spine2_joint (can refine with 3D)

    # Neck & Head
    head = skeleton[0]
    neck_vec = head[:2] - neck[:2]
    neck_angle = np.arctan2(neck_vec[1], neck_vec[0])
    initial_pose[2] = neck_angle    # neck_joint
    initial_pose[3] = 0.0           # head_joint

    # Left Arm
    l_shoulder = skeleton[5]
    l_elbow = skeleton[6]
    l_wrist = skeleton[7]

    upper_arm_vec = l_elbow[:2] - l_shoulder[:2]
    forearm_vec = l_wrist[:2] - l_elbow[:2]

    initial_pose[4] = np.arctan2(upper_arm_vec[1], upper_arm_vec[0])  # left_shoulder_joint
    initial_pose[5] = np.arctan2(forearm_vec[1], forearm_vec[0])      # left_upper_arm_joint
    initial_pose[6] = 0.0                                             # left_forearm_joint
    initial_pose[7] = 0.0                                             # left_hand_joint

    # Right Arm
    r_shoulder = skeleton[2]
    r_elbow = skeleton[3]
    r_wrist = skeleton[4]

    upper_arm_vec = r_elbow[:2] - r_shoulder[:2]
    forearm_vec = r_wrist[:2] - r_elbow[:2]

    initial_pose[8] = np.arctan2(upper_arm_vec[1], upper_arm_vec[0])  # right_shoulder_joint
    initial_pose[9] = np.arctan2(forearm_vec[1], forearm_vec[0])      # right_upper_arm_joint
    initial_pose[10] = 0.0                                           # right_forearm_joint
    initial_pose[11] = 0.0                                           # right_hand_joint

    # Left Leg
    l_hip = skeleton[12]
    l_knee = skeleton[13]
    l_ankle = skeleton[14]

    thigh_vec = l_knee[:2] - l_hip[:2]
    shin_vec = l_ankle[:2] - l_knee[:2]

    initial_pose[12] = np.arctan2(thigh_vec[1], thigh_vec[0])  # left_hip_joint
    initial_pose[13] = np.arctan2(shin_vec[1], shin_vec[0])    # left_thigh_joint
    initial_pose[14] = 0.0                                    # left_shin_joint
    initial_pose[15] = 0.0                                    # left_foot_joint

    # Right Leg
    r_hip = skeleton[9]
    r_knee = skeleton[10]
    r_ankle = skeleton[11]

    thigh_vec = r_knee[:2] - r_hip[:2]
    shin_vec = r_ankle[:2] - r_knee[:2]

    initial_pose[16] = np.arctan2(thigh_vec[1], thigh_vec[0])  # right_hip_joint
    initial_pose[17] = np.arctan2(shin_vec[1], shin_vec[0])    # right_thigh_joint
    initial_pose[18] = 0.0                                    # right_shin_joint
    initial_pose[19] = 0.0                                    # right_foot_joint

    # Optional extra joints (hands/feet, 20-24)
    # Fill zeros or small offsets
    initial_pose[20:] = 0.0

    return initial_pose
