import cv2
import numpy as np


N_POINTS = 100   # number of points in a bounding box
DATA_DIR = "C:/Users/Ana/Desktop/Ana/FER/6.semestar/ZAVRAD/data/"


class MedianFlowTracker(object):

    def __init__(self):
        self.lk_params = dict(winSize=(11, 11),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

    def calculate_next_bounding_box(self, f1, f2, bb1):

        f1bb = cv2.rectangle(f1, bb1, (255, 0, 0), 2)   # frame1 with bounding box drawn on it

        bb1top = bb1[0]
        bb1left = bb1[1]
        bb1height = bb1[2]
        bb1width = bb1[3]

        # todo: napraviti da točkice budu uniformno raspoređene, a ne ovako raštrkane
        points1old = np.empty((N_POINTS, 2))   # matrix with N_POINTS rows and 2 columns
        points1old[:, 0] = np.random.randint(bb1left, bb1left + bb1width, N_POINTS)   # x coordinates of points
        points1old[:, 1] = np.random.randint(bb1top, bb1top + bb1height, N_POINTS)   # y coordinates of points
        points1old = points1old.astype(np.float32)

        f1points_old = self.draw_points_on_frame(f1bb, points1old)
        cv2.imshow("Tracking", f1points_old)
        cv2.waitKey(0)

        points2, st, err = cv2.calcOpticalFlowPyrLK(f1, f2, points1old, None, **self.lk_params)

        f2points = self.draw_points_on_frame(frame2, points2)
        cv2.imshow("Tracking", f2points)
        cv2.waitKey(0)

        points1new, st, err = cv2.calcOpticalFlowPyrLK(f2, f1, points2, None, **self.lk_params)

        f1points_new = self.draw_points_on_frame(f1, points1new)
        cv2.imshow("Tracking", f1points_new)
        cv2.waitKey(0)

        # todo: idući korak je filtrirati 50% točkica


    def draw_points_on_frame(self, frame, points):
        points = points.astype(int)
        red = [0, 0, 255]
        for row in points:
            for i in range(row[0] - 2, row[0] + 2):
                for j in range(row[1] - 2, row[1] + 2):
                    frame[i][j] = red
        return frame


if __name__ == '__main__':

    tracker = MedianFlowTracker()

    # # Read video
    # video = cv2.VideoCapture(DATA_DIR + "walking.mp4")
    #
    # # Exit if video not opened.
    # if not video.isOpened():
    #     print("Could not open video")
    #     sys.exit()
    #
    # # Read the first frame
    # ok, frame1 = video.read()
    # if not ok:
    #     print("Cannot read video file")
    #     sys.exit()
    #
    # # Read the second frame
    # ok, frame2 = video.read()
    # if not ok:
    #     print("Cannot read video file")
    #     sys.exit()

    frame1 = cv2.imread(DATA_DIR + "frame1.png")
    frame2 = cv2.imread(DATA_DIR + "frame2.png")

    # Uncomment the line below to select a different bounding box
    bbox1 = cv2.selectROI(frame1, False)

    tracker.calculate_next_bounding_box(frame1, frame2, bbox1)



