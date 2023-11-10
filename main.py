from ultralytics import YOLO
from utils.utils import (
    move_test_imgs, write_txt_over_img)
import os
import shutil
import glob
import numpy as np
import cv2
import yaml
from PIL import Image
import subprocess
import math


# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
# Configuration
CONFIG_YAML_FILE = 'configuration/configuration.yaml'
with open(CONFIG_YAML_FILE, 'r') as file:
    CONFIG_DATA = yaml.safe_load(file)

# Coordinates of Bounding Boxes
COORDS_FILE_FOLDER = os.path.join(
    CONFIG_DATA['PRJ_PATH'], CONFIG_DATA['DATA_FOLDER_NAME'],
    CONFIG_DATA['SUBJECT'],
    'YOLODataset/labels/test/rect_coords')

# Dataset of images is created by labelme2yolo in this directory
# if the labelme images are manually stored in LABELME_FOLDER
DATASET_FILE = os.path.join(
    CONFIG_DATA['PRJ_PATH'], CONFIG_DATA['DATA_FOLDER_NAME'],
    CONFIG_DATA['SUBJECT'], CONFIG_DATA['DATASET_YAML_FILE'])

# ============================================================================
# CONSTANTS FOR LABELME
# ============================================================================
if CONFIG_DATA['EXECUTION'] == 'labelme2yolo':

    # !!! When labelme is executed, save the files in this directory:
    # "data/brake_caliper_real_time" for example
    LABELME_FOLDER = os.path.join(
        CONFIG_DATA['DATA_FOLDER_NAME'], CONFIG_DATA['SUBJECT'])

# ============================================================================
# CONSTANTS FOR TRAIN
# ============================================================================
elif CONFIG_DATA['EXECUTION'] == 'train':

    # Should be copied in the project path because of its structure
    shutil.copy(DATASET_FILE, CONFIG_DATA['PRJ_PATH'])

    DATASET_FILE_UPDT = os.path.join(
        CONFIG_DATA['PRJ_PATH'], 'dataset.yaml')

# ============================================================================
# CONSTANTS FOR TEST
# ============================================================================
elif CONFIG_DATA['EXECUTION'] == 'test':

    # Should be copied in the project path because of its structure
    shutil.copy(DATASET_FILE, CONFIG_DATA['PRJ_PATH'])

    with open('dataset.yaml', 'r') as file:
        YOLO_DATASET = yaml.safe_load(file)

    TRAIN_IMGS_FOLDER = YOLO_DATASET['train']
    VAL_IMGS_FOLDER = YOLO_DATASET['val']
    TEST_IMGS_FOLDER = YOLO_DATASET['test']
    TEST_IMGS_FOLDER = glob.glob(os.path.join(TEST_IMGS_FOLDER, '*.png'))

    move_test_imgs(
        CONFIG_DATA['DATA_FOLDER_NAME'],
        TRAIN_IMGS_FOLDER, VAL_IMGS_FOLDER, TEST_IMGS_FOLDER)

    BEST_WEIGHTS = os.path.join(
        CONFIG_DATA['TRAIN_RSLT_PATH'], CONFIG_DATA['SUBJECT'],
        CONFIG_DATA['BEST_WEIGHTS_PATH'])

# ============================================================================
# CONSTANTS FOR 2DMATCHING
# ============================================================================
elif CONFIG_DATA['EXECUTION'] == '2dmatching':

    IMG_REF_PATH = os.path.join(
        CONFIG_DATA['IMG_TEMPLATE_PATH'], CONFIG_DATA['TEMPLATE_IMG'])

    if CONFIG_DATA['SELECT_BB_TO_ALIGN']:
        ALGN_IMG_FOLDER = os.path.join(
            CONFIG_DATA['TEST_RSLT_PATH'],
            CONFIG_DATA['NO_BACK_BB_FOLDER_NAME'])
        IMG_REF_NAME = IMG_REF_PATH.split("\\")[-1][:-4] + "_bb"

    else:
        ALGN_IMG_FOLDER = os.path.join(
            CONFIG_DATA['TEST_RSLT_PATH'],
            CONFIG_DATA['NO_BACK_FOLDER_NAME'])
        IMG_REF_NAME = IMG_REF_PATH.split("\\")[-1][:-4]

    ALGN_IMGS_FILES = glob.glob(os.path.join(ALGN_IMG_FOLDER, '*.png'))

    IMG_REF_FOLDER = os.path.join(
        CONFIG_DATA['2D_MATCHING_PATH'],
        IMG_REF_NAME, str(CONFIG_DATA['FEATURES']))

    if not os.path.exists(IMG_REF_FOLDER):
        os.makedirs(IMG_REF_FOLDER)

    IMG_REF_NAME = os.path.join(
        IMG_REF_FOLDER, "AAA_" + IMG_REF_NAME + "_TEMPLATE.jpg")


# ============================================================================
# FUNCTIONS
# ============================================================================
def get_mask(results):
    """
    Returns the masks from YOLO results.
    """

    masks = results[0].masks
    mask = masks[0].data[0].numpy()
    mask_img = Image.fromarray(mask, "I")
    mask_img = mask_img.convert("L")

    return mask_img


def get_no_back(orig_img, mask_img):
    """
    Returns the image without background,
    given the original image PATH and the mask.
    """

    orig_img = Image.open(orig_img)

    # Resize to have the same dimension
    mask_img = mask_img.resize(orig_img.size)

    # Convert the mask to greyscale and then to NumPy array
    mask_array = np.array(mask_img.convert("L"))
    white_mask = mask_array > 200

    # Convert the original to NumPy array
    orig_array = np.array(orig_img)
    intersect_img = np.where(white_mask[:, :, None], orig_array, [0, 0, 0])

    # Create Image object from NumPy array
    no_back_img = Image.fromarray(intersect_img.astype('uint8'))

    # # Convert from Pillow to opencv
    # open_cv_image = np.array(no_back_img)
    # # Convert RGB to BGR
    # open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Close the original images
    orig_img.close()
    mask_img.close()

    return no_back_img


def get_boxes_coords(results):
    """
    Returns the coordinates of the center, the width and the height
    of the Bounding Boxes found by YOLO.
    """

    for r in results:
        for box in r.boxes:

            b = box.xywhn[0]

            width_resized = 1280  # 1280  2064
            height_resized = 720  # 720  1544
            x_cn, y_cn, width_n, height_n = b.tolist()
            xc = x_cn*width_resized
            yc = y_cn*height_resized
            w = width_n*width_resized
            h = height_n*height_resized

            break

    return xc, yc, w, h


def draw_boxes_and_center(src_img, center_x, center_y, width, height):
    """
    Returns the image obtained by drawing the Bounding Box
    over a starting image.
    """
    # ============================= BOXES =============================
    color = (0, 255, 0)
    thickness = 2

    # Calculate the top-left and bottom-right coordinates
    top_left_x = int(center_x - width / 2)
    top_left_y = int(center_y - height / 2)
    bottom_right_x = int(center_x + width / 2)
    bottom_right_y = int(center_y + height / 2)

    # Draw the rectangle on the image
    img_rect = cv2.rectangle(src_img, (top_left_x, top_left_y),
                             (bottom_right_x, bottom_right_y),
                             color,
                             thickness)

    # ============================= CENTER =============================
    # Red X center
    color = (0, 0, 255)
    line_length = 3  # 10  3
    thickness = 2  # 5  2
    # Blue circle center
    raggio = 2  # 5  2
    color2 = (255, 0, 0)

    # Define the center of the rectangle
    center_x = int(center_x)
    center_y = int(center_y)

    # Draw Red X center
    img_rect = cv2.line(
        img_rect, (center_x - line_length, center_y - line_length),
        (center_x + line_length, center_y + line_length), color, thickness)
    img_rect = cv2.line(
        img_rect, (center_x - line_length, center_y + line_length),
        (center_x + line_length, center_y - line_length), color, thickness)
    # Draw Blue circle center
    img_rect = cv2.circle(img_rect, (center_x, center_y), raggio, color2, -1)

    return img_rect


def get_rotated_coords(homography, x, y):
    """
    Returns the original coordinates transformed
    according to the homography matrix.
    """

    # Crea un vettore con il punto iniziale
    point_vec = np.array([[x], [y], [1]])

    # Applica la matrice di omografia al punto
    point_vec_hom = np.dot(homography, point_vec)

    # Estrai le coordinate trasformate
    new_x = point_vec_hom[0, 0] / point_vec_hom[2, 0]
    new_y = point_vec_hom[1, 0] / point_vec_hom[2, 0]

    return new_x, new_y


def rotate_boxes_and_center(
        src_img, homography, center_x, center_y, width, height):
    """
    Returns the original image with the Bounding Boxes
    rotated according to the homography matrix.
    """
    # ============================= BOXES =============================
    color = (0, 255, 0)
    thickness = 2

    # Compute the ORIGINAL coordinates of the top-left and bottom-right
    top_left_x = int(center_x - width / 2)
    top_left_y = int(center_y - height / 2)
    bottom_right_x = int(center_x + width / 2)
    bottom_right_y = int(center_y + height / 2)

    # Compute the ROTATED coordinates of all the vertices
    x_out_top_left, y_out_top_left = get_rotated_coords(
        homography, top_left_x, top_left_y)
    x_out_top_right, y_out_top_right = get_rotated_coords(
        homography, bottom_right_x, top_left_y)
    x_out_bottom_left, y_out_bottom_left = get_rotated_coords(
        homography, top_left_x, bottom_right_y)
    x_out_bottom_right, y_out_bottom_right = get_rotated_coords(
        homography, bottom_right_x, bottom_right_y)

    # Define the coordinates of the ROTATED vertices
    point1 = (int(x_out_top_left), int(y_out_top_left))
    point2 = (int(x_out_top_right), int(y_out_top_right))
    point3 = (int(x_out_bottom_right), int(y_out_bottom_right))
    point4 = (int(x_out_bottom_left), int(y_out_bottom_left))

    # Slope of the line that connects the ROTATED bottom vertices
    # The ORIGINAL slope is 0

    # TOL is used to avoid computations of number --> inf
    TOL = 0.5

    if abs(point4[0] - point3[0]) > TOL:
        slope_rotated = \
            (abs(point4[1] - point3[1])) / (abs(point4[0] - point3[0]))
        # Get the angle of the rotated BB
        angle_slope = math.degrees(math.atan(-slope_rotated))
        # 1st quarter
        if (point3[0] > point4[0]) and (point3[1] < point4[1]):
            angle_slope = -(angle_slope)
        # 2nd quarter
        if (point3[0] < point4[0]) and (point3[1] < point4[1]):
            angle_slope = 180 - abs(angle_slope)
        # 3rd quarter
        elif (point3[0] < point4[0]) and (point3[1] > point4[1]):
            angle_slope = 180 + abs(angle_slope)
        # 4th quarter
        elif (point3[0] > point4[0]) and (point3[1] > point4[1]):
            angle_slope = 360 - abs(angle_slope)
    else:
        # Angle = pi/2: point4[0] - point3[0] = 0
        if point3[1] < point4[1]:  # 90°
            angle_slope = 90
        elif point3[1] > point4[1]:  # 270°
            angle_slope = 270

    # Draw the lines that connect the vertices
    img_rot_rect = cv2.line(src_img, point1, point2, color, thickness)
    img_rot_rect = cv2.line(src_img, point2, point3, color, thickness)
    img_rot_rect = cv2.line(src_img, point3, point4, color, thickness)
    img_rot_rect = cv2.line(src_img, point4, point1, color, thickness)

    # ============================= CENTER =============================
    # Red X center
    color = (0, 0, 255)
    line_length = 3  # 10  3
    thickness = 2  # 5  2
    # Blue circle center
    raggio = 2  # 5  2
    color2 = (255, 0, 0)

    # Define the center of the rectangle
    center_x = int(center_x)
    center_y = int(center_y)

    # Compute the vertices of the ORIGINAL X lines
    line1_start = (center_x - line_length, center_y - line_length)
    line1_end = (center_x + line_length, center_y + line_length)
    line2_start = (center_x - line_length, center_y + line_length)
    line2_end = (center_x + line_length, center_y - line_length)

    # Compute the vertices of the ROTATED X lines
    x_line1_start, y_line1_start = get_rotated_coords(
        homography, line1_start[0], line1_start[1])
    x_line1_end, y_line1_end = get_rotated_coords(
        homography, line1_end[0], line1_end[1])
    x_line2_start, y_line2_start = get_rotated_coords(
        homography, line2_start[0], line2_start[1])
    x_line2_end, y_line2_end = get_rotated_coords(
        homography, line2_end[0], line2_end[1])

    # Define the coordinates of the ROTATED X vertices
    line1_start = (int(x_line1_start), int(y_line1_start))
    line1_end = (int(x_line1_end), int(y_line1_end))
    line2_start = (int(x_line2_start), int(y_line2_start))
    line2_end = (int(x_line2_end), int(y_line2_end))

    # Draw Red X center
    img_rot_rect = cv2.line(
        img_rot_rect, line1_start, line1_end, color, thickness)
    img_rot_rect = cv2.line(
        img_rot_rect, line2_start, line2_end, color, thickness)
    # Draw Blue circle center
    img_rot_rect = cv2.circle(
        img_rot_rect, (center_x, center_y), raggio, color2, -1)

    return img_rot_rect, angle_slope


if __name__ == '__main__':

    # ========================================================================
    # LABELME
    # ========================================================================
    if CONFIG_DATA['EXECUTION'] == 'labelme':

        try:
            subprocess.run([CONFIG_DATA['EXECUTION']], check=False)
        except subprocess.CalledProcessError as e:
            print(f"Errore durante l'esecuzione di labelme: {e}")

    # ========================================================================
    # LABELME2YOLO
    # ========================================================================
    if CONFIG_DATA['EXECUTION'] == 'labelme2yolo':

        try:
            subprocess.run(
                [CONFIG_DATA['EXECUTION'],
                 "--json_dir",
                 LABELME_FOLDER,
                 "--val_size", "0.01"], check=False)

        except subprocess.CalledProcessError as e:
            print(f"Errore durante l'esecuzione di labelme2yolo: {e}")

    # ========================================================================
    # TRAIN
    # ========================================================================
    if CONFIG_DATA['EXECUTION'] == 'train':

        try:
            # load a pretrained model
            model = YOLO('yolov8n-seg.pt')

            results = model.train(
                data=DATASET_FILE_UPDT, epochs=100,
                patience=20, imgsz=1280,
                project=CONFIG_DATA['TRAIN_RSLT_PATH'],
                name=CONFIG_DATA['SUBJECT'], exist_ok=True)

        except Exception as e:
            print(f"Errore durante l'addestramento del modello: {e}")

    # ========================================================================
    # TEST
    # ========================================================================
    if CONFIG_DATA['EXECUTION'] == 'test':

        model = YOLO(BEST_WEIGHTS)

        coords_to_txt = []

        for file in TEST_IMGS_FOLDER:

            results = model.predict(
                file, conf=CONFIG_DATA['PREDICT_CONFIDENCE'], save=True,
                project=CONFIG_DATA['TEST_RSLT_PATH'],
                exist_ok=True, imgsz=1280)

            # If there are detections...
            if results[0].masks is not None:

                if CONFIG_DATA['DO_IMG_PROCESSING']:

                    # Get YOLO masks
                    mask_img = get_mask(results)
                    no_back_img = get_no_back(file, mask_img)

                    # Convert from Pillow to opencv
                    open_cv_image = np.array(no_back_img)
                    # Convert RGB to BGR
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    cv2.imshow("ciao", open_cv_image)
                    cv2.waitKey(0)

                    # Define save directories
                    filename = file.split("\\")[-1]
                    mask_path = no_back_path = os.path.join(
                        CONFIG_DATA['TEST_RSLT_PATH'],
                        CONFIG_DATA['MASK_FOLDER_NAME'], filename)
                    no_back_path = os.path.join(
                        CONFIG_DATA['TEST_RSLT_PATH'],
                        CONFIG_DATA['NO_BACK_FOLDER_NAME'], filename)
                    no_back_bb_path = os.path.join(
                        CONFIG_DATA['TEST_RSLT_PATH'],
                        CONFIG_DATA['NO_BACK_BB_FOLDER_NAME'], filename)

                    # Save and check
                    try:
                        mask_img.save(mask_path, "PNG")
                    except FileNotFoundError as e:
                        print(f"Errore: {e}. File non trovato: {mask_path}")
                    try:
                        no_back_img.save(no_back_path, "PNG")
                    except FileNotFoundError as e:
                        print(f"Errore: {e}. File non trovato: {no_back_path}")

                    # Convert the image without background to cv2
                    no_back_img = cv2.imread(no_back_path)

                    # Get the bounding box coordinates
                    xc, yc, w, h = get_boxes_coords(results)

                    # Draw the bounding box over the image without background
                    rect_draw = draw_boxes_and_center(
                        no_back_img, xc, yc, w, h)

                    coords_to_txt.append((filename, xc, yc, w, h))

                    cv2.imwrite(no_back_bb_path, rect_draw)

            else:
                print("=== No detections! === "
                      "Try to lower the PREDICT_CONFIDENCE parameter ===")

        # Write the bounding boxes coordinates in a txt file
        with open(COORDS_FILE_FOLDER, "w") as file:
            for coord in coords_to_txt:
                file.write(
                    f"{coord[0]} {coord[1]} {coord[2]} "
                    f"{coord[3]} {coord[4]}\n")

    # ========================================================================
    # 2D MATCHING
    # ========================================================================
    elif CONFIG_DATA['EXECUTION'] == '2dmatching':

        # Store the template image
        IMG_TEMPLATE = cv2.imread(IMG_REF_PATH)

        if CONFIG_DATA['DO_RESIZE']:
            # Resize the template image
            IMG_TEMPLATE = cv2.resize(
                IMG_TEMPLATE,
                (CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT']))

        # Convert the template image to BGR
        IMG_TEMPLATE_BGR = cv2.cvtColor(IMG_TEMPLATE, cv2.COLOR_BGR2GRAY)
        # Store the dimensions of BGR image
        HEIGHT, WIDTH = IMG_TEMPLATE_BGR.shape

        cv2.imwrite(IMG_REF_NAME, IMG_TEMPLATE.astype(np.uint8))

        # Extract template box coordinates
        with open(COORDS_FILE_FOLDER, "r") as txtfile:
            for line in txtfile:
                if CONFIG_DATA['TEMPLATE_IMG'] in line:
                    xc_template, yc_template, w_template, h_template =\
                        map(float, line.split()[1:])
                break

        # Start a loop to align all the images without background
        for file in ALGN_IMGS_FILES:

            # Exclude the template image
            if CONFIG_DATA['TEMPLATE_IMG'] not in file:

                img_to_align = cv2.imread(file)

                if CONFIG_DATA['DO_RESIZE']:
                    img_to_align = cv2.resize(
                        img_to_align,
                        (CONFIG_DATA['RSZ_WIDTH'], CONFIG_DATA['RSZ_HEIGHT']))

                # Convert to grayscale
                img_algn_BGR = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

                # ===========================================================
                # ORB DETECTOR (with BRUTE-FORCE matcher)
                # ===========================================================
                if CONFIG_DATA['DETECTOR'] == "ORB":
                    # Create ORB detector with features
                    orb_detector = cv2.ORB_create(CONFIG_DATA['FEATURES'])
                    # Find keypoints and descriptors
                    kp1, d1 = orb_detector.detectAndCompute(
                        img_algn_BGR, None)
                    kp2, d2 = orb_detector.detectAndCompute(
                        IMG_TEMPLATE_BGR, None)

                    if CONFIG_DATA['MATCHER'] == "BRUTE-FORCE":
                        matcher = cv2.BFMatcher(
                            cv2.NORM_HAMMING, crossCheck=True)

                # ===========================================================
                # SIFT DETECTOR (with either BRUTE-FORCE or FLANN matcher)
                # ===========================================================
                elif CONFIG_DATA['DETECTOR'] == "SIFT":
                    sift = cv2.SIFT_create()
                    kp1, d1 = sift.detectAndCompute(img_algn_BGR, None)
                    kp2, d2 = sift.detectAndCompute(IMG_TEMPLATE_BGR, None)

                    if CONFIG_DATA['MATCHER'] == "BRUTE-FORCE":
                        matcher = cv2.BFMatcher()

                    elif CONFIG_DATA['MATCHER'] == "FLANN":
                        FLAN_INDEX_KDTREE = 0
                        index_params = dict(
                            algorithm=FLAN_INDEX_KDTREE, trees=5)
                        search_params = dict(checks=50)
                        matcher = cv2.FlannBasedMatcher(
                            index_params, search_params)
                        matches = matcher.knnMatch(d1, d2, k=2)
                        good_matches = []
                        for m1, m2 in matches:
                            if m1.distance < 0.5 * m2.distance:
                                good_matches.append([m1])
                        flann_matches = cv2.drawMatchesKnn(
                            img_algn_BGR, kp1, IMG_TEMPLATE_BGR, kp2,
                            good_matches, None, flags=2)

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
                homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
                # Inverse homography: from template to 'to_align'
                inv_homography, inv_mask = cv2.findHomography(
                    p2, p1, cv2.RANSAC)

                # 'img_to_align' aligned to template
                transformed_img = cv2.warpPerspective(
                    img_to_align, homography, (WIDTH, HEIGHT))

                # Elements extraction from homography matrix
                rot1 = homography[0, 0]
                rot2 = homography[0, 1]
                to_align_center_x, to_align_center_y = get_rotated_coords(
                    inv_homography, xc_template, yc_template)

                # Compute angle of rotation in rad
                angle_rad = np.arctan2(rot2, rot1)
                # Convert from rad to deg
                angle_deg = np.degrees(angle_rad)

                # =========================================================
                # Save results
                img_name = file.split("\\")[-1][:-4]

                # 'img_to_align' with inverse rotated bounding box
                rotated_img_bb = rotate_boxes_and_center(
                    img_to_align, inv_homography,
                    xc_template, yc_template, w_template, h_template)

                txt_to_write = "CENTER: " + \
                    f"({to_align_center_x}, {to_align_center_y})." +\
                    f"  ANGLE: {-angle_deg} degrees"

                rotated_img_bb = write_txt_over_img(
                    rotated_img_bb, txt_to_write)

                cv2.imwrite(os.path.join(
                    IMG_REF_FOLDER,
                    img_name + "_1_BEFORE.jpg"),
                    rotated_img_bb)

                # 'aligned_image' with template bounding box
                rotated_img_bb_aligned = draw_boxes_and_center(
                    transformed_img,
                    xc_template, yc_template, w_template, h_template)

                txt_to_write = f"CENTER: ({xc_template}, {yc_template})"

                rotated_img_bb_aligned = write_txt_over_img(
                    rotated_img_bb_aligned, txt_to_write)

                cv2.imwrite(os.path.join(
                    IMG_REF_FOLDER,
                    img_name + "_" +
                    f"{CONFIG_DATA['FEATURES']}" + "_2_AFTER.jpg"),
                    rotated_img_bb_aligned)

                # =========================================================
                # Print info
                print(f"*** IMAGE {img_name} HAS BEEN ROTATED "
                      f"OF {angle_deg} DEGREES ***")
                print(f"*** dx = {to_align_center_x - xc_template} ***")
                print(f"*** dy = {to_align_center_y - yc_template} ***\n")
