import cv2
import numpy as np
import argparse
import os

class PoseEstimatorModule1:
    """
    Module 1: Pose Estimation & Initial Pose Extraction (OpenPose BODY_25)
    - Multi-person detection
    - Target selection (largest bounding box)
    - Full-body kinematic conversion (joint angles)
    - Output annotated image + initial pose vector θ_init
    """

    def __init__(self, proto_path='./models/pose_deploy.prototxt', model_path='./models/pose_iter_584000.caffemodel'):
        # BODY_25 Keypoint Mapping
        self.BODY_25_MAP = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
            "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
            "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
            "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
            "RHeel": 24, "Background": 25
        }

        self.POSE_PAIRS = [
            (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
            (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
            (14, 21), (14, 19), (19, 20), (11, 24), (11, 22), (22, 23)
        ]

        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found at {proto_path} or {model_path}")

        self.net = cv2.dnn.readNet(model_path, proto_path)
        print("="*75)
        print("OpenPose BODY_25 model loaded successfully.")
        print("="*75)

    def process_image(self, image_path):
        # ----- Task 1.1: Image Acquisition and Preprocessing ----- #
        image = cv2.imread(image_path)
        if image is None:
            print("="*75)
            print(f"Could not read image at {image_path}")
            print("="*75)
            return None, None
        
        # ----- Task 1.2: Multi-Person 25-Keypoint Extraction ----- #
        skeletons = self._extract_skeletons(image)
        if len(skeletons) == 0:
            print("No skeletons detected.")
            return image, None
        print(f"Skeletons:\n{skeletons}")

        # ----- Task 1.3: Target Skeleton Selection ----- #
        target_skeleton = self._select_target_skeleton(skeletons)
        if target_skeleton is None:
            print("Target skeleton selection failed.")
            return image, None

        # ----- Task 1.4: Kinematic Conversion ----- #
        theta_init = self._convert_keypoints_to_angles(target_skeleton)

        # ----- Visualization ----- #
        annotated = self._draw_skeleton(image.copy(), target_skeleton)

        return annotated, theta_init

    # Skeleton Extraction 
    def _extract_skeletons(self, image):
        img_h, img_w, _ = image.shape
        in_w, in_h = 368, 368

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (in_w, in_h), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        n_points = len(self.BODY_25_MAP) - 1
        heatmaps = output[0, :n_points, :, :]

        # Simple multi-person detection placeholder
        # Full OpenPose PAF logic needed for real multi-person extraction
        skeletons = []
        skeleton = np.zeros((n_points, 3), dtype=np.float32)

        for i in range(n_points):
            heatmap = cv2.resize(heatmaps[i], (img_w, img_h))
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            if conf > 0.1:
                skeleton[i] = (point[0], point[1], conf)
            else:
                skeleton[i] = (0, 0, 0)

        if np.any(skeleton[:, 2] > 0):
            skeletons.append(skeleton)

        return skeletons

    # Target Selection 
    def _select_target_skeleton(self, skeletons):
        largest_area = -1
        target = None
        for skel in skeletons:
            valid = skel[skel[:, 2] > 0]
            if valid.shape[0] < 4:
                continue
            min_x, max_x = np.min(valid[:, 0]), np.max(valid[:, 0])
            min_y, max_y = np.min(valid[:, 1]), np.max(valid[:, 1])
            area = (max_x - min_x) * (max_y - min_y)
            if area > largest_area:
                largest_area = area
                target = skel
        return target

    # Kinematic Conversion
    def _convert_keypoints_to_angles(self, skeleton):
        def get_point(name):
            idx = self.BODY_25_MAP[name]
            if skeleton[idx, 2] > 0:
                return skeleton[idx, :2]
            return None

        def vector_angle(a, b):
            """2D angle from a to b in radians"""
            return np.arctan2(b[1] - a[1], b[0] - a[0])

        theta = []

        # --- Torso ---
        mid_hip = get_point("MidHip")
        neck = get_point("Neck")
        nose = get_point("Nose")

        spine_angle = vector_angle(mid_hip, neck) if mid_hip is not None and neck is not None else 0.0
        theta.append(spine_angle * 0.5)  # spine1_joint
        theta.append(spine_angle * 0.5)  # spine2_joint

        neck_angle = vector_angle(neck, nose) if neck is not None and nose is not None else 0.0
        theta.append(neck_angle)  # neck_joint
        theta.append(neck_angle)  # head_joint

        # --- Left Arm ---
        l_shoulder = get_point("LShoulder")
        l_elbow = get_point("LElbow")
        l_wrist = get_point("LWrist")
        l_hand = l_wrist  # approximation

        shoulder_angle = vector_angle(l_shoulder, l_elbow) if l_shoulder is not None and l_elbow is not None else 0.0
        upper_arm_angle = vector_angle(l_shoulder, l_elbow) if l_shoulder is not None and l_elbow is not None else 0.0
        forearm_angle = (vector_angle(l_elbow, l_wrist) - shoulder_angle) if l_elbow is not None and l_wrist is not None else 0.0
        hand_angle = vector_angle(l_wrist, l_hand) if l_wrist is not None and l_hand is not None else 0.0

        theta.extend([shoulder_angle, upper_arm_angle, forearm_angle, hand_angle])

        # --- Right Arm ---
        r_shoulder = get_point("RShoulder")
        r_elbow = get_point("RElbow")
        r_wrist = get_point("RWrist")
        r_hand = r_wrist  # approximation

        shoulder_angle = vector_angle(r_shoulder, r_elbow) if r_shoulder is not None and r_elbow is not None else 0.0
        upper_arm_angle = vector_angle(r_shoulder, r_elbow) if r_shoulder is not None and r_elbow is not None else 0.0
        forearm_angle = (vector_angle(r_elbow, r_wrist) - shoulder_angle) if r_elbow is not None and r_wrist is not None else 0.0
        hand_angle = vector_angle(r_wrist, r_hand) if r_wrist is not None and r_hand is not None else 0.0

        theta.extend([shoulder_angle, upper_arm_angle, forearm_angle, hand_angle])

        # --- Left Leg ---
        l_hip = get_point("LHip")
        l_knee = get_point("LKnee")
        l_ankle = get_point("LAnkle")
        l_foot = l_ankle  # approximation

        hip_angle = vector_angle(l_hip, l_knee) if l_hip is not None and l_knee is not None else 0.0
        thigh_angle = hip_angle
        shin_angle = (vector_angle(l_knee, l_ankle) - hip_angle) if l_knee is not None and l_ankle is not None else 0.0
        foot_angle = vector_angle(l_ankle, l_foot) if l_ankle is not None and l_foot is not None else 0.0

        theta.extend([hip_angle, thigh_angle, shin_angle, foot_angle])

        # --- Right Leg ---
        r_hip = get_point("RHip")
        r_knee = get_point("RKnee")
        r_ankle = get_point("RAnkle")
        r_foot = r_ankle  # approximation

        hip_angle = vector_angle(r_hip, r_knee) if r_hip is not None and r_knee is not None else 0.0
        thigh_angle = hip_angle
        shin_angle = (vector_angle(r_knee, r_ankle) - hip_angle) if r_knee is not None and r_ankle is not None else 0.0
        foot_angle = vector_angle(r_ankle, r_foot) if r_ankle is not None and r_foot is not None else 0.0

        theta.extend([hip_angle, thigh_angle, shin_angle, foot_angle])

        return np.array(theta, dtype=np.float32)


    # Visualization 
    def _draw_skeleton(self, image, skeleton):
        for i in range(skeleton.shape[0]):
            x, y, c = skeleton[i]
            if c > 0:
                cv2.circle(image, (int(x), int(y)), 3, (245, 117, 66), -1) 

        for pair in self.POSE_PAIRS:
            a, b = pair
            if skeleton[a, 2] > 0 and skeleton[b, 2] > 0:
                pt1 = tuple(skeleton[a, :2].astype(int))
                pt2 = tuple(skeleton[b, :2].astype(int))
                cv2.line(image, pt1, pt2, (245, 66, 230), 1) 

        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Module 1: Pose Estimation & Initial Pose Extraction')
    parser.add_argument('--image', required=True, type=str, help='Path to the input image file')
    args = parser.parse_args()

    try:
        estimator = PoseEstimatorModule1()
        annotated, theta_init = estimator.process_image(args.image)

        if theta_init is not None:
            print("="*75)
            print("Initial Pose Vector (θ_init) in Radians:")
            np.set_printoptions(precision=4, suppress=True)
            print(theta_init)
            print("="*75)

            filename = os.path.basename(args.image)
            name = os.path.splitext(filename)[0]
            out_dir = os.path.join("./output_images", name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"openpose_{filename}")
            cv2.imwrite(out_path, annotated)
            print(f"Output image saved to: {out_path}")
            print("="*75)
        else:
            print("\nPose Extraction Failed.")
            print("="*75)

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
