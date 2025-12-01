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

     # Task: Multi-Person 25-Keypoint Extraction
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

    # Task: Select largest person
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
    def extract_keypoints(self,image):
        
        results = self.detector.process((image * 255).astype(np.uint8))
        # print("Mediapipe pose landmarks: ", results.pose_landmarks)
        if not results.pose_landmarks:
            return []
        
        landmarks = results.pose_landmarks.landmark
        keypoints_33 = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])

        def meanval(i, j):
            p1, p2 = keypoints_33[i], keypoints_33[j]
            return (p1 + p2) / 2.0
        
        #body_25 points
        body25 = [
            keypoints_33[0],            # 0 - Nose
            meanval(11, 12),                # 1 - Neck
            keypoints_33[12],           # 2 - RShoulder
            keypoints_33[14],           # 3 - RElbow
            keypoints_33[16],           # 4 - RWrist
            keypoints_33[11],           # 5 - LShoulder
            keypoints_33[13],           # 6 - LElbow
            keypoints_33[15],           # 7 - LWrist
            meanval(23, 24),                # 8 - MidHip
            keypoints_33[24],           # 9 - RHip
            keypoints_33[26],           # 10 - RKnee
            keypoints_33[28],           # 11 - RAnkle
            keypoints_33[23],           # 12 - LHip
            keypoints_33[25],           # 13 - LKnee
            keypoints_33[27],           # 14 - LAnkle
            keypoints_33[2],            # 15 - REye
            keypoints_33[5],            # 16 - LEye
            keypoints_33[8],            # 17 - REar
            keypoints_33[7],            # 18 - LEar
            keypoints_33[32],           # 19 - LBigToe
            keypoints_33[31],           # 20 - LSmallToe
            keypoints_33[29],           # 21 - LHeel
            keypoints_33[28],           # 22 - RBigToe
            keypoints_33[27],           # 23 - RSmallToe
            keypoints_33[30]            # 24 - RHeel
        ]

        body25 = np.array(body25)
        return body25
    
    def draw_skeleton(self, image, skeleton_points, save_path):
        """
        Draw body 25-like skeleton on top of the original image (not black background).
        """
        # Convert float RGB image [0,1] to uint8 BGR [0,255] for OpenCV drawing
        output_image = cv2.cvtColor((image.copy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        h, w, _ = output_image.shape

        # Define approximate BODY25-like connections (with spine added)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Right arm
            (1, 5), (5, 6), (6, 7),               # Left arm
            (1, 8), (8, 9), (9, 10), (10, 11),    # Right leg
            (8, 12), (12, 13), (13, 14),          # Left leg
            (0, 15), (15, 17),                    # Right head
            (0, 16), (16, 18),                    # Left head
            (1, 8)                                # Spine (neck → hip)
        ]

        # Drawing the skeleton connections in blue color
        for start, end in connections:
            x1, y1, z1, c1 = skeleton_points[start]
            x2, y2, z2, c2 = skeleton_points[end]
            if c1 > 0 and c2 > 0:
                cv2.line(output_image, (int(x1 * w), int(y1 * h)),
                        (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)  # Blue

        # Annotate the joints as circles in green color with index labels
        for i, (x, y, z, c) in enumerate(skeleton_points):
            if c > 0:
                cv2.circle(output_image, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)  # Green
                cv2.putText(output_image, str(i), (int(x * w) + 4, int(y * h) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        
        if save_path:
            cv2.imwrite(save_path, output_image)
            print(f"Saved the skeleton diagram at: {save_path}")
        else:
            cv2.imshow("Skeleton over the image", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()