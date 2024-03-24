import cv2
import numpy as np


def slam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Example values
    fx = 640
    fy = 640
    cx = 320
    cy = 240

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp_map, des_map = None, None
    prev_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        if kp_map is None or des_map is None:
            kp_map, des_map = kp_frame, des_frame
            prev_frame = frame
            continue

        min_kp_num = min(len(kp_map), len(kp_frame))
        kp_map = kp_map[:min_kp_num]
        kp_frame = kp_frame[:min_kp_num]
        des_map = des_map[:min_kp_num]
        des_frame = des_frame[:min_kp_num]

        matches = bf.match(des_map, des_frame)

        pts1 = np.float32(
            [kp_map[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        pts2 = np.float32(
            [kp_frame[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T)

        if K is not None:
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            _, _, _, _, _, R, t = cv2.solvePnPRansac(
                points_3d, pts2, K, None, criteria)

        for point in points_3d:
            x, y, z = point[0]
            cv2.circle(prev_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        cv2.imshow('SLAM', prev_frame)
        prev_frame = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    slam()
