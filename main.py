import cv2
import numpy as np
from PIL import Image, ImageDraw


PIXELS_BETWEEN_POINTS = 5
DATA_DIR = "C:/Users/Ana/Desktop/Ana/FER/6.semestar/ZAVRAD/data/"


class MedianFlowTracker(object):

    def __init__(self):
        self.lk_params = dict(winSize=(11, 11),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

    def calculate_next_bounding_box(self, frame_1, frame_2, bounding_box_1):

        # DRAW A BOUNDING BOX ON FRAME 1
        frame_1_copy = np.copy(frame_1)
        frame_1_with_bounding_box = cv2.rectangle(frame_1_copy, bounding_box_1, (255, 0, 0), 2)
        cv2.imshow("Tracking", frame_1_with_bounding_box)
        cv2.waitKey(0)

        # CALCULATE FORWARD OPTICAL FLOW
        frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
        frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)

        flow = None
        flow = cv2.calcOpticalFlowFarneback(prev=frame_1_gray, next=frame_2_gray, flow=flow, pyr_scale=0.8,
                                            levels=15, winsize=5, iterations=10, poly_n=5, poly_sigma=0, flags=10)
        # magnitude, radian_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # pi = 22 / 7
        # degree_angle = [rad*(180/pi) for rad in radian_angle]
        h, w = frame_2.shape[:2]
        step = 16
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        fx_median = np.median(fx)
        fy_median = np.median(fy)

        print(fx_median)
        print(fy_median)

        # MOVE BOUNDING BOX
        bounding_box_2 = (int(bounding_box_1[0] + fx_median),
                          int(bounding_box_1[1] + fy_median),
                          int(bounding_box_1[2]),
                          int(bounding_box_1[3]))

        # CUT BOUNDING BOX 2 IF OUTSIDE OF FRAME 2
        bounding_box_2 = (min(bounding_box_2[0], frame_2.shape[1]),
                          min(bounding_box_2[1], frame_2.shape[0]),
                          min(bounding_box_2[2], frame_2.shape[1]),
                          min(bounding_box_2[3], frame_2.shape[0]))

        # DRAW A BOUNDING BOX ON FRAME 2
        frame_2_copy = np.copy(frame_2)
        frame_2_with_bounding_box = cv2.rectangle(frame_2_copy, bounding_box_2, (255, 0, 0), 2)
        cv2.imshow("Tracking", frame_2_with_bounding_box)
        cv2.waitKey(0)


if __name__ == '__main__':
    tracker = MedianFlowTracker()

    frame1 = cv2.imread(DATA_DIR + "walking1.png")
    frame2 = cv2.imread(DATA_DIR + "walking2.png")

    bbox1 = cv2.selectROI(frame1, False)
    cv2.destroyAllWindows()

    tracker.calculate_next_bounding_box(frame1, frame2, bbox1)
