import cv2
import numpy as np


def prepare_frame(frame1, frame2):
    frame = cv2.merge([frame1, frame2])
    return frame


def get_merged_frame_pair(vc, ob, settings):
    vc.set(cv2.CAP_PROP_POS_FRAMES, int(ob['prev_video_frame']))
    ret, frame1 = vc.read()
    ret, frame2 = vc.read()
    if settings['masked']:
        frame1 = mask_frame(frame1, ob, settings, prev=True)
        frame2 = mask_frame(frame2, ob, settings)
    if settings['scrolling']:
        # get scrolling offsets between observed frames
        off_x = int(ob.get("scroll_offset_x"))
        off_y = int(ob.get("scroll_offset_y"))

        # crop frames according to scrolling offsets
        height = frame1.shape[0]
        width = frame1.shape[1]
        empty1 = np.zeros(frame1.shape, frame1.dtype)
        empty2 = np.zeros(frame1.shape, frame1.dtype)
        frame1 = frame1[max(0, off_y):min(height, (height + off_y)),
                       max(0, off_x):min(width, (width + off_x))]
        frame2 = frame2[max(0, -off_y):min(height, (height - off_y)),
                       max(0, -off_x):min(width, (width - off_x))]
        empty1[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
        empty2[0:frame1.shape[0], 0:frame1.shape[1]] = frame2
        frame1 = empty1
        frame2 = empty2
    return prepare_frame(frame1, frame2)


def mask_frame(frame, ob, settings, prev=False):
    mask_name = ob.get("webpage") + "_view_mask_" + ob.get("observation_id")
    if prev:
        mask_name += "_prev"
    mask_name += ".png"
    mask_file = settings['data_folder'] / "Dataset_visual_change" / ob.get('user') / mask_name
    mask = cv2.imread(str(mask_file), 0)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

