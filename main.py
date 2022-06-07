import cv2
import numpy as np

PIXELS_BETWEEN_POINTS = 5
DATA_DIR = "data/"
FILE_NAME = "walking.mp4"
## CHOOSE ONE OF THE FOLLOWING: walking.mp4, cars.mp4, skating.mp4, optimist.mp4, red.mp4, driving.mp4



class MedianFlowTracker(object):

    @staticmethod
    def calculate_next_bounding_box(prev, curr, bb):

        lk_params = dict(winSize=(11, 11),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        _n_samples = 100
        _fb_max_dist = 1
        _ds_factor = 0.95
        _min_n_points = 10

        # sample points inside the bounding box
        p0 = np.empty((_n_samples, 2))
        p0[:, 0] = np.random.randint(bb[0], bb[0] + bb[2] + 1, _n_samples)
        p0[:, 1] = np.random.randint(bb[1], bb[1] + bb[3] + 1, _n_samples)

        p0 = p0.astype(np.float32)

        # forward-backward tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)
        indx = np.where(st == 1)[0]
        p0 = p0[indx, :]
        p1 = p1[indx, :]
        p0r, st, err = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None, **lk_params)
        if err is None:
            return None

        # check forward-backward error and min number of points
        fb_dist = np.abs(p0 - p0r).max(axis=1)
        good = fb_dist < _fb_max_dist

        # keep half of the points
        err = err[good].flatten()
        if len(err) < _min_n_points:
            return None

        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        p0 = (p0[good])[half_indx]
        p1 = (p1[good])[half_indx]

        # estimate displacement
        dx = np.median(p1[:, 0] - p0[:, 0])
        dy = np.median(p1[:, 1] - p0[:, 1])

        # all pairs in prev and curr
        i, j = np.triu_indices(len(p0), k=1)
        pdiff0 = p0[i] - p0[j]
        pdiff1 = p1[i] - p1[j]

        # estimate change in scale
        p0_dist = np.sum(pdiff0 ** 2, axis=1)
        p1_dist = np.sum(pdiff1 ** 2, axis=1)
        ds = np.sqrt(np.median(p1_dist / (p0_dist + 2 ** -23)))
        ds = (1.0 - _ds_factor) + _ds_factor * ds

        # MOVE BOUNDING BOX
        bounding_box_width_change = bb[2] #* (x_resize - 1)
        bounding_box_height_change = bb[3] #* (y_resize - 1)
        bounding_box_2_left = int(bb[0] + dx - bounding_box_width_change / 2 + 0.5)
        bounding_box_2_top = int(bb[1] + dy - bounding_box_height_change / 2 + 0.5)
        bounding_box_2_width = int(bb[2]) #* x_resize + 0.2)
        bounding_box_2_height = int(bb[3]) #* y_resize + 0.2)
        bounding_box_2 = (bounding_box_2_left, bounding_box_2_top, bounding_box_2_width, bounding_box_2_height)

        # CUT BOUNDING BOX 2 IF OUTSIDE OF FRAME 2
        bounding_box_2 = (min(bounding_box_2[0], curr.shape[1]),
                          min(bounding_box_2[1], curr.shape[0]),
                          min(bounding_box_2[2], curr.shape[1]),
                          min(bounding_box_2[3], curr.shape[0]))

        return bounding_box_2


if __name__ == '__main__':

    tracker = MedianFlowTracker()

    video = cv2.VideoCapture(DATA_DIR + FILE_NAME)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # SELECT BOUNDING BOX ON THE FIRST FRAME
    init_ret, init_frame = video.read()
    bbox2 = cv2.selectROI(init_frame, False)
    bounding_box_minimum_width = bbox2[2] * 0.25
    bounding_box_minimum_height = bbox2[3] * 0.25
    cv2.destroyAllWindows()

    ret2, frame2 = video.read()
    while True:
        ret1, frame1, bbox1 = ret2, frame2, bbox2
        ret2, frame2 = video.read()
        if ret1 and ret2:
            bbox2 = tracker.calculate_next_bounding_box(frame1, frame2, bbox1)
            cv2.rectangle(frame2, bbox2, (255, 0, 0), 2)
            cv2.imshow("", frame2)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    video.release()