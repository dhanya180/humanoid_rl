import sys
import os
import cv2
import numpy as np
import argparse

# ------------------ Step 1: Add pyopenpose from another env ------------------
pyopenpose_path = "<your path_to_pyopenpose>" 
if pyopenpose_path not in sys.path:
    sys.path.append(pyopenpose_path)

os.environ['LD_LIBRARY_PATH'] = "<your path_openpose_libraries>" + os.environ.get('LD_LIBRARY_PATH', '')

try:
    from openpose import pyopenpose as op
except ImportError as e:
    print("Cannot import pyopenpose. Check pyopenpose path and LD_LIBRARY_PATH.")
    raise e

# ------------------ BODY_25 Keypoint Mapping ------------------
BODY_25_MAP = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
    "RHeel": 24
}

# ------------------ OpenPose Wrapper ------------------
class PoseEstimator:
    def __init__(self, model_folder="your_path_to_openpose_models"):
        params = {
            "model_folder": model_folder,
            "model_pose": "BODY_25",
            "net_resolution": "-1x368",
            "number_people_max": -1
        }
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        print("OpenPose BODY_25 API initialized successfully.")

    # Task 1.1: Image Acquisition & Preprocessing
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return image

    # Task 1.2: Multi-Person 25-Keypoint Extraction
    def detect_skeletons(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        datums = op.VectorDatum()
        datums.append(datum)
        self.opWrapper.emplaceAndPop(datums)

        keypoints = datum.poseKeypoints  # (nPersons, 25, 3)
        if keypoints is None or len(keypoints.shape) < 3:
            print("No persons detected.")
            return [], image
        print(f"Detected {keypoints.shape[0]} person(s).")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Keypoints data:\n{keypoints}")
        return keypoints, image

    # Task 1.3: Select largest person
    def select_target_skeleton(self, keypoints):
        largest_area = -1
        target_idx = 0
        for i, person in enumerate(keypoints):
            valid = person[person[:, 2] > 0]
            if valid.shape[0] < 4:
                continue
            x_min, x_max = valid[:, 0].min(), valid[:, 0].max()
            y_min, y_max = valid[:, 1].min(), valid[:, 1].max()
            area = (x_max - x_min) * (y_max - y_min)
            if area > largest_area:
                largest_area = area
                target_idx = i
        return keypoints[target_idx]

    # Convert keypoints to humanoid θ_init
    def keypoints_to_angles(self, skeleton):
        def get_point(name):
            idx = BODY_25_MAP[name]
            return skeleton[idx, :2] if skeleton[idx, 2] > 0 else None

        def vector_angle(a, b):
            if a is None or b is None:
                return 0.0
            return np.arctan2(b[1] - a[1], b[0] - a[0])

        theta = []

        # Torso
        mid_hip = get_point("MidHip")
        neck = get_point("Neck")
        nose = get_point("Nose")
        spine_angle = vector_angle(mid_hip, neck)
        theta.extend([spine_angle * 0.5, spine_angle * 0.5])
        neck_angle = vector_angle(neck, nose)
        theta.extend([neck_angle, neck_angle])

        # Left Arm
        l_shoulder, l_elbow, l_wrist = get_point("LShoulder"), get_point("LElbow"), get_point("LWrist")
        shoulder_angle = vector_angle(l_shoulder, l_elbow)
        forearm_angle = vector_angle(l_elbow, l_wrist) - shoulder_angle
        theta.extend([shoulder_angle, shoulder_angle, forearm_angle, forearm_angle])

        # Right Arm
        r_shoulder, r_elbow, r_wrist = get_point("RShoulder"), get_point("RElbow"), get_point("RWrist")
        shoulder_angle = vector_angle(r_shoulder, r_elbow)
        forearm_angle = vector_angle(r_elbow, r_wrist) - shoulder_angle
        theta.extend([shoulder_angle, shoulder_angle, forearm_angle, forearm_angle])

        # Left Leg
        l_hip, l_knee, l_ankle = get_point("LHip"), get_point("LKnee"), get_point("LAnkle")
        hip_angle = vector_angle(l_hip, l_knee)
        shin_angle = vector_angle(l_knee, l_ankle) - hip_angle
        theta.extend([hip_angle, hip_angle, shin_angle, shin_angle])

        # Right Leg
        r_hip, r_knee, r_ankle = get_point("RHip"), get_point("RKnee"), get_point("RAnkle")
        hip_angle = vector_angle(r_hip, r_knee)
        shin_angle = vector_angle(r_knee, r_ankle) - hip_angle
        theta.extend([hip_angle, hip_angle, shin_angle, shin_angle])

        return np.array(theta, dtype=np.float32)

    # Draw only the target person's skeleton
    def annotate_target(self, image, skeleton):
        for joint_idx in range(25):
            if skeleton[joint_idx, 2] > 0:
                cv2.circle(image, tuple(skeleton[joint_idx, :2].astype(int)), 4, (0, 255, 0), -1)

        # Draw bones
        connections = [
            (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
            (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
            (14, 21), (14, 19), (19, 20), (11, 24), (11, 22), (22, 23)
        ]
        for a, b in connections:
            if skeleton[a, 2] > 0 and skeleton[b, 2] > 0:
                pt1 = tuple(skeleton[a, :2].astype(int))
                pt2 = tuple(skeleton[b, :2].astype(int))
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        return image


# ------------------ Main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select one person and compute θ_init")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    args = parser.parse_args()

    estimator = PoseEstimator()

    # Load image
    image = estimator.load_image(args.image)

    # Detect skeletons
    keypoints, _ = estimator.detect_skeletons(image)

    if len(keypoints) > 0:
        # Select one person
        target_skeleton = estimator.select_target_skeleton(keypoints)
        # θ_init for simulation
        theta_init = estimator.keypoints_to_angles(target_skeleton)
        print("Initial Pose Vector (θ_init):")
        np.set_printoptions(precision=4, suppress=True)
        print(theta_init)

        # Annotate only the selected person
        annotated_image = estimator.annotate_target(image.copy(), target_skeleton)
    else:
        theta_init = None
        annotated_image = image
        print("No skeleton selected.")

    # Save annotated image
    out_dir = os.path.join("./output_images", os.path.splitext(os.path.basename(args.image))[0])
    os.makedirs(out_dir, exist_ok=True)
    out_img_path = os.path.join(out_dir, f"target_openpose_{os.path.basename(args.image)}")
    cv2.imwrite(out_img_path, annotated_image)
    print(f"Annotated image saved to: {out_img_path}")
