import threading
import cv2
from typing import Optional
from vimba import (Vimba, Camera,
                   VimbaCameraError,
                   VimbaFeatureError,
                   Frame,
                   FrameStatus,
                   intersect_pixel_formats,
                   OPENCV_PIXEL_FORMATS,
                   COLOR_PIXEL_FORMATS,
                   MONO_PIXEL_FORMATS)
from utils.utils import write_txt_over_img
from ultralytics import YOLO
import os
import yaml
from main import (
    get_rotated_coords, rotate_boxes_and_center,
    draw_boxes_and_center)
import numpy as np
import time


def get_mask(results):
    """
    Returns the masks from YOLO results using OpenCV.
    """

    masks = results[0].masks
    mask = masks[0].data[0].numpy()

    # Converte la maschera in un'immagine in scala di grigi utilizzando OpenCV
    mask_img = (mask * 255).astype(np.uint8)

    return mask_img


def get_no_back(src_img, mask_img):
    """
    Returns the image without background,
    given the original image and the mask.
    """

    # src_img = src_img.as_opencv_image()

    # Resize the mask to have the same dimension as the original image
    mask_img = cv2.resize(mask_img, (src_img.shape[1], src_img.shape[0]))

    # Convert the mask to binary (threshold) image
    _, mask_binary = cv2.threshold(mask_img, 200, 255, cv2.THRESH_BINARY)

    # Apply the mask to the original image
    no_back_img = cv2.bitwise_and(src_img, src_img, mask=mask_binary)

    return no_back_img


def get_boxes_coords(results, width_resized, height_resized):
    """
    Returns the coordinates of the center, the width and the height
    of the Bounding Boxes found by YOLO.
    """

    for r in results:
        for box in r.boxes:

            b = box.xywhn[0]

            x_cn, y_cn, width_n, height_n = b.tolist()
            xc = x_cn*width_resized
            yc = y_cn*height_resized
            w = width_n*width_resized
            h = height_n*height_resized

            break

    return xc, yc, w, h


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                print(
                    'Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                print('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera):
    with cam:

        cam.ExposureAuto.set('Off')
        cam.ExposureTime.set(CONFIG_DATA['EXPOSURE'])
        cam.Gain.set(CONFIG_DATA['GAIN'])
        cam.Gamma.set(CONFIG_DATA['GAMMA'])
        cam.Saturation.set(CONFIG_DATA['SATURATION'])
        cam.BalanceWhiteAuto.set('Off')
        cam.ConvolutionMode.set('Sharpness')
        cam.Sharpness.set(CONFIG_DATA['SHARPNESS'])

        # Try to adjust GeV packet size. This Feature is only available
        # for GigE - Cameras.
        try:
            cam.GVSPAdjustPacketSize.run()
            while not cam.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VimbaFeatureError):
            pass
        # Query available, open_cv compatible pixel formats
        # prefer color formats over monochrome formats
        cv_fmts = intersect_pixel_formats(
            cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
        color_fmts = intersect_pixel_formats(
            cv_fmts, COLOR_PIXEL_FORMATS)
        if color_fmts:
            cam.set_pixel_format(color_fmts[0])
        else:
            mono_fmts = intersect_pixel_formats(
                cv_fmts, MONO_PIXEL_FORMATS)

            if mono_fmts:
                cam.set_pixel_format(mono_fmts[0])


CONFIG_YAML_FILE = 'configuration/configuration.yaml'
with open(CONFIG_YAML_FILE, 'r') as file:
    CONFIG_DATA = yaml.safe_load(file)

CAMERA_ID = CONFIG_DATA['CAMERA_ID']

BEST_WEIGHTS = os.path.join(
    CONFIG_DATA['TRAIN_RSLT_PATH'], CONFIG_DATA['SUBJECT'],
    CONFIG_DATA['BEST_WEIGHTS_PATH'])

# Take the template image with its coordinates
# Coordinates of Bounding Boxes
COORDS_FILE_FOLDER = os.path.join(
    CONFIG_DATA['PRJ_PATH'], CONFIG_DATA['DATA_FOLDER_NAME'],
    CONFIG_DATA['SUBJECT'],
    'YOLODataset/labels/test/rect_coords')

with open('dataset.yaml', 'r') as file:
    YOLO_DATASET = yaml.safe_load(file)

TEMPLATE_IMG_FILE = os.path.join(
    YOLO_DATASET['train'], CONFIG_DATA['TEMPLATE_IMG'])

model = YOLO(BEST_WEIGHTS)
results = model.predict(
    TEMPLATE_IMG_FILE,
    conf=CONFIG_DATA['PREDICT_CONFIDENCE'], save=False,
    project=CONFIG_DATA['TEST_RSLT_PATH'],
    exist_ok=True, imgsz=CONFIG_DATA['YOLO_IMGSZ'])
xc, yc, w, h = get_boxes_coords(
    results, CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT'])
str_title = CONFIG_DATA['TEMPLATE_IMG']

with open(COORDS_FILE_FOLDER, "w") as file:
    file.write(
        f"{str_title} {xc} {yc} "
        f"{w} {h}\n")

IMG_REF_PATH = os.path.join(
    CONFIG_DATA['IMG_TEMPLATE_PATH'], CONFIG_DATA['TEMPLATE_IMG'])

IMG_TEMPLATE = cv2.imread(IMG_REF_PATH)

if CONFIG_DATA['DO_TEMPLATE_RESIZE']:
    # Resize the template image
    IMG_TEMPLATE = cv2.resize(
        IMG_TEMPLATE, (CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT']))

# Convert the template image to BGR
IMG_TEMPLATE_BGR = cv2.cvtColor(IMG_TEMPLATE, cv2.COLOR_BGR2GRAY)
# Store the dimensions of BGR image
HEIGHT, WIDTH = IMG_TEMPLATE_BGR.shape

# Extract template box coordinates
with open(COORDS_FILE_FOLDER, "r") as txtfile:
    for line in txtfile:
        if CONFIG_DATA['TEMPLATE_IMG'] in line:
            xc_template, yc_template, w_template, h_template =\
                map(float, line.split()[1:])
        break

# Create ORB detector with features
orb_detector = cv2.ORB_create(CONFIG_DATA['FEATURES'])
kp2, d2 = orb_detector.detectAndCompute(IMG_TEMPLATE_BGR, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class Handler:

    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):

        ENTER_KEY_CODE = 13
        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:

            print('{} acquired {}'.format(cam, frame), flush=True)

            start_time = time.time()

            # 1) YOLO test
            results = model.predict(
                frame.as_opencv_image(),
                conf=CONFIG_DATA['PREDICT_CONFIDENCE'],
                save=False, imgsz=CONFIG_DATA['YOLO_IMGSZ'])

            frame_rsz = cv2.resize(
                frame.as_opencv_image(),
                (CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT']))

            print(frame_rsz.shape)

            # If there are detections...
            if results[0].masks is not None:

                # 2) Remove background
                # Get YOLO masks
                mask_img = get_mask(results)
                mask_img = cv2.resize(
                    mask_img,
                    (CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT']))
                no_back_img = get_no_back(frame_rsz, mask_img)
                no_back_img = cv2.resize(
                    no_back_img,
                    (CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT']))
                # Convert from Pillow to opencv
                no_back_img = np.array(no_back_img)
                # Convert RGB to BGR
                no_back_img = no_back_img[:, :, ::-1].copy()

                # 3) 2D matching with the static template
                img_algn_BGR = cv2.cvtColor(no_back_img, cv2.COLOR_BGR2GRAY)

                # Find keypoints and descriptors
                kp1, d1 = orb_detector.detectAndCompute(img_algn_BGR, None)

                # Match the two sets of descriptors.
                matches = matcher.match(d1, d2)

                # Convert the tuple into a list
                matches_list = list(matches)
                # Sort the list according to Hamming distance
                matches_list.sort(key=lambda x: x.distance)
                # Convert the list into a tuple
                matches = tuple(matches_list)

                # Take the top 90% matches
                matches = matches[:int(len(matches)*0.9)]
                no_of_matches = len(matches)

                # Define empty matrices of shape [no_of_matches * 2]
                p1 = np.zeros((no_of_matches, 2))
                p2 = np.zeros((no_of_matches, 2))

                # Extract the coordinates of the keypoints in the images
                for i in range(len(matches)):
                    p1[i, :] = kp1[matches[i].queryIdx].pt
                    p2[i, :] = kp2[matches[i].trainIdx].pt

                # Find the homography matrix
                homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
                # Inverse homography: from template to 'to_align'
                inv_homography, _ = cv2.findHomography(
                    p2, p1, cv2.RANSAC)

                """
                # Projection vector
                homography[2][0] = 10e-5
                homography[2][1] = 10e-5
                inv_homography[2][0] = 10e-5
                inv_homography[2][1] = 10e-5

                # Scaling
                homography[1][0] = 0.5
                homography[1][1] = 0.5
                inv_homography[1][0] = -0.5
                inv_homography[1][1] = 0.5

                print(homography)
                print(inv_homography)
                """

                # Elements extraction from homography matrix
                to_align_center_x, to_align_center_y = get_rotated_coords(
                    inv_homography, xc_template, yc_template)

                transformed_img = cv2.warpPerspective(
                    no_back_img, homography, (WIDTH, HEIGHT))

                rotated_img_bb_aligned = draw_boxes_and_center(
                    transformed_img,
                    xc_template, yc_template, w_template, h_template)

                cv2.namedWindow(
                    "IMAGE ROTATED TO TEMPLATE POSITION", cv2.WINDOW_NORMAL)
                cv2.imshow("IMAGE ROTATED TO TEMPLATE POSITION",
                           rotated_img_bb_aligned)

                rotated_img_bb, angle_slope = rotate_boxes_and_center(
                    no_back_img, inv_homography,
                    xc_template, yc_template, w_template, h_template)

                cv2.namedWindow("BOUNDING BOX", cv2.WINDOW_NORMAL)
                cv2.imshow("BOUNDING BOX", rotated_img_bb)

                end_time = time.time()
                frame_rate = 1/(end_time-start_time)

                # 4) Show the text
                # Add text to show angle and coordinates
                txt_to_write = "CENTER: " + \
                    f"({round(to_align_center_x, 3)}, " + \
                    f"{round(to_align_center_y, 3)})." + \
                    f"  ANGLE: {round(angle_slope, 3)}" + \
                    f"  FRAME RATE: {round(frame_rate, 3)} frame/s"

                write_txt_over_img(
                    frame_rsz, txt_to_write,
                    to_align_center_x, to_align_center_y)

                # Display the resulting frame
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame_rsz)

            else:
                # Display the resulting frame
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame_rsz)

        cam.queue_frame(frame)


def main():

    with Vimba.get_instance():
        with get_camera(CAMERA_ID) as cam:

            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler()

            try:
                # Start Streaming with a custom a buffer of 10 Frames
                # (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=3)
                handler.shutdown_event.wait()

            finally:
                cam.stop_streaming()


main()
