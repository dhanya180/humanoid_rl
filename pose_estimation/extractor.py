import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

class PoseEstimatorModule1:
    """
    Module 1: Pose Estimation
    - Detects human pose keypoints using MediaPipe Pose
    - Converts landmarks to joint angles (θ_init)
    - Draws pose skeleton on image
    """
    def __init__(self):
        """Initialize MediaPipe Pose model and drawing utilities"""
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,  # Single image mode
            model_complexity=2       # High accuracy model
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_image(self, image_path):
        """
        Process an image to extract pose landmarks and joint angles.
        Returns the annotated image and the initial pose vector.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read the image file at {image_path}")
            return None, None

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Run pose detection
        results = self.pose_detector.process(image_rgb)
        if not results.pose_landmarks:
            print("Could not detect any pose landmarks in the image.")
            return image, None

        # Convert landmarks to joint angles
        initial_pose_vector = self._convert_landmarks_to_angles(results.pose_landmarks, w, h)

        # Draw landmarks on image
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        return image, initial_pose_vector

    def _convert_landmarks_to_angles(self, landmarks, img_width, img_height):
        """
        Convert 2D pose landmarks to joint angles (right/left hip and knee).
        Returns a NumPy array: [right_hip, right_knee, left_hip, left_knee]
        """
        lm = landmarks.landmark

        def get_coords(idx):
            p = lm[idx]
            return np.array([p.x * img_width, p.y * img_height])

        # Left leg joints
        left_hip = get_coords(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        left_knee = get_coords(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_ankle = get_coords(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)

        # Right leg joints
        right_hip = get_coords(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_knee = get_coords(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_ankle = get_coords(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        # Vectors between joints
        v_l_hip_knee = left_knee - left_hip
        v_l_knee_ankle = left_ankle - left_knee
        v_r_hip_knee = right_knee - right_hip
        v_r_knee_ankle = right_ankle - right_knee

        # Calculate angles relative to vertical
        left_hip_angle = np.arctan2(v_l_hip_knee[0], v_l_hip_knee[1])
        left_knee_angle = np.arctan2(v_l_knee_ankle[0], v_l_knee_ankle[1]) - left_hip_angle
        right_hip_angle = np.arctan2(v_r_hip_knee[0], v_r_hip_knee[1])
        right_knee_angle = np.arctan2(v_r_knee_ankle[0], v_r_knee_ankle[1]) - right_hip_angle

        # Assemble pose vector and clip knee angles to [0, 2.5] radians
        pose_vector = np.array([right_hip_angle, right_knee_angle, left_hip_angle, left_knee_angle])
        pose_vector[1] = np.clip(pose_vector[1], 0, 2.5)
        pose_vector[3] = np.clip(pose_vector[3], 0, 2.5)

        return pose_vector

    def close(self):
        """Release MediaPipe resources"""
        self.pose_detector.close()


if __name__ == '__main__':
    # Parse input image path
    parser = argparse.ArgumentParser(description='Module 1: Pose Estimation and Joint Angle Extraction')
    parser.add_argument('--image', required=True, type=str, help='Path to the input image file')
    args = parser.parse_args()

    # Create Pose Estimator
    estimator = PoseEstimatorModule1()

    # Process image
    processed_image, initial_pose = estimator.process_image(args.image)

    if initial_pose is not None:
        print("\n✅ Pose Extraction Successful!")
        print("--------------------------------------------------")
        print("Final Initial Pose Vector (θ_init):")
        np.set_printoptions(precision=4, suppress=True)
        print(initial_pose)
        print("--------------------------------------------------")

        # Prepare output folder: ./output_images/<image_name>/
        input_filename = os.path.basename(args.image)
        image_name = os.path.splitext(input_filename)[0]
        output_dir = os.path.join("./output_images", image_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, input_filename)

        # Save processed image
        cv2.imwrite(output_path, processed_image)
        print(f"Output image saved to: {output_path}")
    else:
        print("\n❌ Pose Extraction Failed. Please check the image file and path.")

    # Clean up
    estimator.close()
