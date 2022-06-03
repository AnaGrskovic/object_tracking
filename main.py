import cv2
import numpy as np
from PIL import Image, ImageDraw


PIXELS_BETWEEN_POINTS = 5
DATA_DIR = "C:/Users/Ana/Desktop/Ana/FER/6.semestar/ZAVRAD/data/"


class MedianFlowTracker(object):

    def calculate_next_bounding_box(self, frame_1, frame_2, bounding_box_1):

        bounding_box_1_left = bounding_box_1[0]
        bounding_box_1_top = bounding_box_1[1]
        bounding_box_1_width = bounding_box_1[2]
        bounding_box_1_height = bounding_box_1[3]
        cropped_frame_1 = frame_1[bounding_box_1_top: bounding_box_1_top + bounding_box_1_height,
                          bounding_box_1_left: bounding_box_1_left + bounding_box_1_width]
        cropped_frame_2 = frame_2[bounding_box_1_top: bounding_box_1_top + bounding_box_1_height,
                          bounding_box_1_left: bounding_box_1_left + bounding_box_1_width]

        # CALCULATE OPTICAL FLOW
        cropped_gray_frame_1 = cv2.cvtColor(cropped_frame_1, cv2.COLOR_RGB2GRAY)
        cropped_gray_frame_2 = cv2.cvtColor(cropped_frame_2, cv2.COLOR_RGB2GRAY)

        flow_forward = None
        flow_forward = cv2.calcOpticalFlowFarneback(prev=cropped_gray_frame_1, next=cropped_gray_frame_2,
                                                    flow=flow_forward, pyr_scale=0.8,
                                                    levels=15, winsize=5, iterations=10, poly_n=5, poly_sigma=0,
                                                    flags=10)
        flow_backward = None
        flow_backward = cv2.calcOpticalFlowFarneback(prev=cropped_gray_frame_2, next=cropped_gray_frame_1,
                                                     flow=flow_backward, pyr_scale=0.8,
                                                     levels=15, winsize=5, iterations=10, poly_n=5, poly_sigma=0,
                                                     flags=10)

        # FILTER OUT THE SMALLEST FORWARD BACKWARD ERROR
        flow_diff = np.add(flow_forward, flow_backward)
        flow_diff = np.abs(flow_diff)
        flow_diff_array = np.array(flow_diff).flatten()
        flow_diff_x = flow_diff_array[::2]
        flow_diff_y = flow_diff_array[1::2]
        flow_diff_x_median = np.median(flow_diff_x)
        flow_diff_y_median = np.median(flow_diff_y)

        best_x = flow_diff_x < flow_diff_x_median
        best_y = flow_diff_y < flow_diff_y_median

        counter_x = 0
        best_indices_x = []
        for i in best_x:
            if i:
                best_indices_x.append(counter_x)
            counter_x += 1
        counter_y = 0
        best_indices_y = []
        for i in best_y:
            if i:
                best_indices_y.append(counter_y)
            counter_y += 1

        overall_good_indices = best_indices_x + best_indices_y
        overall_best_indices = [value for value in best_indices_x if value in best_indices_y]

        flow_forward_array = np.array(flow_forward).flatten()
        flow_forward_best = [flow_forward_array[i * 2:i * 2 + 2] for i in overall_best_indices]
        flow_backward_array = np.array(flow_backward).flatten()
        flow_backward_best = [flow_backward_array[i * 2:i * 2 + 2] for i in overall_best_indices]

        # CALCULATE MOVEMENT
        x_movement = [elem[0] for elem in flow_forward_best]
        y_movement = [elem[1] for elem in flow_forward_best]
        x_movement_real_median = np.median(x_movement)
        y_movement_real_median = np.median(y_movement)
        print(x_movement_real_median)
        print(y_movement_real_median)
        # here i have decided to take a bigger movement than the median
        sorted_x_movement = sorted(x_movement, key=abs)
        sorted_y_movement = sorted(y_movement, key=abs)
        chosen_index = round(0.9 * len(sorted_x_movement))
        x_movement_median = sorted_x_movement[chosen_index]
        y_movement_median = sorted_y_movement[chosen_index]
        print(x_movement_median)
        print(y_movement_median)

        # MOVE BOUNDING BOX
        bounding_box_2 = (int(bounding_box_1[0] + x_movement_median),
                          int(bounding_box_1[1] + y_movement_median),
                          int(bounding_box_1[2]),
                          int(bounding_box_1[3]))

        # CUT BOUNDING BOX 2 IF OUTSIDE OF FRAME 2
        bounding_box_2 = (min(bounding_box_2[0], frame_2.shape[1]),
                          min(bounding_box_2[1], frame_2.shape[0]),
                          min(bounding_box_2[2], frame_2.shape[1]),
                          min(bounding_box_2[3], frame_2.shape[0]))

        return bounding_box_2


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
