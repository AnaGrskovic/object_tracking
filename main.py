import cv2
import numpy as np

PIXELS_BETWEEN_POINTS = 5
DATA_DIR = "C:/Users/Ana/Desktop/Ana/FER/6.semestar/ZAVRAD/data/"


class MedianFlowTracker(object):

    def __init__(self):

    def calculate_next_bounding_box(self, frame_1, frame_2, bounding_box_1):


    def draw_points_on_frame(self, frame, points):
        points = points.astype(int)
        red = [0, 0, 255]
        for row in points:
            for i in range(row[0] - 1, row[0] + 1):
                for j in range(row[1] - 1, row[1] + 1):
                    try:
                        frame[i][j] = red
                    except IndexError:
                        pass
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

    frame1 = cv2.imread(DATA_DIR + "birds3.png")
    frame2 = cv2.imread(DATA_DIR + "birds4.png")

    # Uncomment the line below to select a different bounding box
    bbox1 = cv2.selectROI(frame1, False)
    cv2.destroyAllWindows()

    tracker.calculate_next_bounding_box(frame1, frame2, bbox1)
