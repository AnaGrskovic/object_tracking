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

        # CALCULATE FORWARD OPTICAL FLOW
        points_new_2, st, err = cv2.calcOpticalFlowPyrLK(frame_1, frame_2, points_old_1, None, **self.lk_params)

        # CALCULATE BACKWARD OPTICAL FLOW
        points_new_1, st, err = cv2.calcOpticalFlowPyrLK(frame_2, frame_1, points_new_2, None, **self.lk_params)

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

        # CALCULATE DISPLACEMENT ON X AND Y AXIS
        delta_x = np.median(points_best_2[:, 0] - points_best_1[:, 0])
        delta_y = np.median(points_best_2[:, 1] - points_best_1[:, 1])

        # CALCULATE CHANGE IN SCALE
        # todo

        # MOVE BOUNDING BOX
        bounding_box_2 = (int(bounding_box_1[0] + delta_x),
                          int(bounding_box_1[1] + delta_y),
                          int(bounding_box_1[2]),
                          int(bounding_box_1[3]))

        # CUT BOUNDING BOX 2 IF OUTSIDE OF FRAME 2
        bounding_box_2 = (min(bounding_box_2[0], frame_2.shape[1]),
                          min(bounding_box_2[1], frame_2.shape[0]),
                          min(bounding_box_2[2], frame_2.shape[1]),
                          min(bounding_box_2[3], frame_2.shape[0]))

        return bounding_box_2


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

    video = cv2.VideoCapture(DATA_DIR + "walking.mp4")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # SELECT BOUNDING BOX ON THE FIRST FRAME
    init_frame = cv2.imread(DATA_DIR + "walking_init.png")
    dim = (width, height)
    init_frame = cv2.resize(init_frame, dim, interpolation=cv2.INTER_AREA)
    bbox2 = cv2.selectROI(init_frame, False)
    cv2.destroyAllWindows()

    # PLAY VIDEO
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MPEG'), fps, (height, width))
    ret2, frame2 = video.read()

    while True:
        ret1, frame1, bbox1 = ret2, frame2, bbox2
        ret2, frame2 = video.read()
        if ret1 and ret2:
            bbox2 = tracker.calculate_next_bounding_box(frame1, frame2, bbox1)
            cv2.rectangle(frame2, bbox2, (255, 0, 0), 2)
            output.write(frame2)
            cv2.imshow("", frame2)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    output.release()
    video.release()
