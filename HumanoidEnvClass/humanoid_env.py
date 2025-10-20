import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class HumanoidWalkEnv(gym.Env):
    def __init__(self, urdf_path=r"assets\urdf\humanoid_25dof.urdf", gui=True, time_step=1./40.):
        super(HumanoidWalkEnv, self).__init__()

        # PyBullet setup
        self.gui = gui
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        self.time_step = time_step

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load humanoid URDF
        self.urdf_path = urdf_path
        self.robot_id = p.loadURDF(self.urdf_path, [0,0,1], useFixedBase=False)

        # Define joints
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = [i for i in range(self.num_joints)]
        self.joint_limits = [p.getJointInfo(self.robot_id, i)[8:10] for i in self.joint_indices]

        # Observation space: joint angles + joint velocities + base pos/orientation
        obs_high = np.array([np.finfo(np.float32).max]* (self.num_joints*2 + 6))
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action space: discretized joint torque changes (-1, 0, 1) per joint
        self.action_space = spaces.MultiDiscrete([3]*self.num_joints)

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = np.array([s[0] for s in joint_states])
        joint_velocities = np.array([s[1] for s in joint_states])
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        obs = np.concatenate([joint_positions, joint_velocities, base_pos, base_lin_vel])
        return obs.astype(np.float32)

    def reset(self, initial_pose=None):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path, [0,0,1], useFixedBase=False)

        if initial_pose is not None:
            for i, angle in enumerate(initial_pose):
                if i >= self.num_joints:
                    break
                p.resetJointState(self.robot_id, i, angle)

        obs = self._get_observation()
        base_pos,_ = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, _ = p.getBaseVelocity(self.robot_id)

        info = {
            "torso_pos": np.array(base_pos),
            "torso_lin_vel": np.array(base_lin_vel)
        }
        return obs, info

    def step(self, action):
        # Map discrete action (0,1,2) to torque: -1,0,1
        torques = np.array(action) - 1.0
        max_torque = 50  # can be tuned
        torques *= max_torque

        # Apply torques
        for i, torque in enumerate(torques):
            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                    jointIndex=i,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=torque)

        p.stepSimulation()
        if self.gui:
            time.sleep(self.time_step)

        obs = self._get_observation()


        # Reward: forward progress along X axis - small penalty for deviation/falling
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, _ = p.getBaseVelocity(self.robot_id)

        info = {
            "torso_pos": np.array(base_pos),
            "torso_lin_vel": np.array(base_lin_vel)
        }
        
        x_pos = base_pos[0]
        reward = x_pos

        # Termination: if humanoid falls (base height too low or tipped)
        done = False
        if base_pos[2] < 0.5:  # height threshold
            done = True
            reward -= 10.0  # penalty for falling

        return obs, reward, done, False, info

    def close(self):
        p.disconnect()
