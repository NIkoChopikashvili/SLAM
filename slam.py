import cv2
import numpy as np


def slam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp_map, des_map = None, None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break
        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        if kp_map is None or des_map is None:
            kp_map, des_map = kp_frame, des_frame
            continue

        matches = bf.match(des_map, des_frame)

        img_matches = cv2.drawMatches(
            frame, kp_map, frame, kp_frame, matches, None)

        cv2.imshow('SLAM', img_matches)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    slam()
