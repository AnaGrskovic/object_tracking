import cv2
import numpy as np

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

        # INITIALIZE A GRID OF POINTS
        bounding_box_1_left = bounding_box_1[0]
        bounding_box_1_top = bounding_box_1[1]
        bounding_box_1_width = bounding_box_1[2]
        bounding_box_1_height = bounding_box_1[3]

        point_x_range = []
        i = bounding_box_1_left
        while i < (bounding_box_1_left + bounding_box_1_width):
            point_x_range.append(i)
            i += PIXELS_BETWEEN_POINTS
        point_y_range = []
        i = bounding_box_1_top
        while i < (bounding_box_1_top + bounding_box_1_height):
            point_y_range.append(i)
            i += PIXELS_BETWEEN_POINTS

        point_x_coords, point_y_coords = np.meshgrid(point_x_range, point_y_range)

        point_x_coords = point_x_coords.reshape((np.prod(point_x_coords.shape),))
        point_y_coords = point_y_coords.reshape((np.prod(point_y_coords.shape),))

        number_of_points = len(point_x_range) * len(point_y_range)

        points_old_1 = np.empty((number_of_points, 2))  # matrix with N_POINTS rows and 2 columns
        points_old_1[:, 0] = point_y_coords
        points_old_1[:, 1] = point_x_coords
        points_old_1 = points_old_1.astype(np.float32)

        # DRAW AN ORIGINAL GRID OF POINTS ON FRAME 1
        frame_1_with_bounding_box_copy = np.copy(frame_1_with_bounding_box)
        frame_1_with_old_points = self.draw_points_on_frame(frame_1_with_bounding_box_copy, points_old_1)
        cv2.imshow("Tracking", frame_1_with_old_points)
        cv2.waitKey(0)

        # CALCULATE FORWARD OPTICAL FLOW
        points_new_2, st, err = cv2.calcOpticalFlowPyrLK(frame_1, frame_2, points_old_1, None, **self.lk_params)

        # DRAW A CALCULATED GRID OF POINTS ON FRAME 2
        frame_2_copy = np.copy(frame_2)
        frame_2_with_new_points = self.draw_points_on_frame(frame_2_copy, points_new_2)
        cv2.imshow("Tracking", frame_2_with_new_points)
        cv2.waitKey(0)

        # CALCULATE BACKWARD OPTICAL FLOW
        points_new_1, st, err = cv2.calcOpticalFlowPyrLK(frame_2, frame_1, points_new_2, None, **self.lk_params)

        # DRAW A CALCULATED GRID OF POINTS ON FRAME 1
        frame_1_copy = np.copy(frame_1)
        frame_1_with_new_points = self.draw_points_on_frame(frame_1_copy, points_new_1)
        cv2.imshow("Tracking", frame_1_with_new_points)
        cv2.waitKey(0)

        # FILTER OUT HALF OF POINTS WITH THE SMALLEST FORWARD BACKWARD ERROR
        fb_distances = np.abs(points_old_1 - points_new_1).max(axis=1)
        distances_median = np.median(fb_distances)

        best = fb_distances < distances_median  # true if point is in the better half, false otherwise
        counter = 0
        best_indices = []
        for i in best:
            if i:
                best_indices.append(counter)
            counter += 1

        points_best_1 = [points_new_1[i] for i in best_indices]
        points_best_1 = np.stack(points_best_1, axis=0)

        points_best_2 = [points_new_2[i] for i in best_indices]
        points_best_2 = np.stack(points_best_2, axis=0)

        # DRAW A BEST GRID OF POINTS ON FRAME 2
        frame_2_copy = np.copy(frame_2)
        frame_2_with_best_points = self.draw_points_on_frame(frame_2_copy, points_best_2)
        cv2.imshow("Tracking", frame_2_with_best_points)
        cv2.waitKey(0)


    def draw_points_on_frame(self, frame, points):
        points = points.astype(int)
        red = [0, 0, 255]
        for row in points:
            for i in range(row[0] - 1, row[0] + 1):
                for j in range(row[1] - 1, row[1] + 1):
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
