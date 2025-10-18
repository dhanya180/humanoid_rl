# HumanoidRL 🤖🚶‍♂️

A project to train a simulated humanoid robot to walk using Deep Reinforcement Learning. The robot learns to start walking from a variety of initial poses extracted directly from real-world images of people.

## 💡 About The Project

The project pipeline consists of three main stages:
* **Perception (Pose Estimation)**: A computer vision module takes a static image, detects people, and extracts a skeletal pose. This pose is converted into a vector of joint angles.
* **Simulation (Physics Environment)**: A PyBullet environment loads a humanoid model and sets its initial stance based on the angles from the perception stage..
* **Control (DRL Agent)**: A Deep Q-Network (DQN) agent learns to control the humanoid's joints to achieve stable, forward walking by maximizing a reward function.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

This project is developed on a Linux (Ubuntu) system and uses Conda for environment management.

### Installation

1.  **Clone the repository**
    ```sh
    git clone https://github.com/dhanya180/humanoid_rl.git
    cd humanoid_rl
    ```

2.  **Create and Activate a Conda Environment**
    ```sh
    conda create --name humanoid_rl python=3.10 -y
    conda activate humanoid_rl
    ```

3.  **Install Python Packages**
    ```sh
    pip install -r requirements.txt
    ```
---

## ▶️ Usage

### Running the Pose Estimation Module

You can run the pose estimation module as a standalone script to extract a humanoid's starting pose from an image.

1.  Run the script by providing the path to your image:
    ```sh
    cd pose_estimation
    python extractor.py --image sample_images/<your_image>.jpg
    ```

2.  **Expected Output:**
    * The script will print whether a person was detected in the image.
    * It will save the processed image with the skeletal overlay in a folder structure based on the input image (e.g., ./output_images/4/4.jpg).
    * The final joint angle vector ($\theta_{init}$) will be printed in the terminal for the detected person.
