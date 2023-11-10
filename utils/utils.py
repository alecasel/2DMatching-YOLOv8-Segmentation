import cv2
import os
import shutil


def move_test_imgs(
        all_imgs_path, train_imgs_folder, val_imgs_folder, test_imgs_folder):
    """
    Create a copy of all the images in the right folders
    """

    # Filename of all the images without file extension
    all_imgs = [os.path.splitext(img)[0]
                for img in os.listdir(all_imgs_path)
                if img.endswith('.jpg')]

    # Filename of the images in train and val without file extension
    train_imgs = [os.path.splitext(img)[0]
                  for img in os.listdir(train_imgs_folder)
                  if img.endswith('.png')]
    val_imgs = [os.path.splitext(img)[0]
                for img in os.listdir(val_imgs_folder)
                if img.endswith('.png')]

    # Keep 'all_imgs' that are neither in 'train_imgs' nor in 'val_imgs'
    test_imgs = [nome for nome in all_imgs
                 if nome not in train_imgs
                 and nome not in val_imgs]

    # Move 'test_imgs' to the right folder
    for nome in test_imgs:
        img_file_path = os.path.join(all_imgs_path, f"{nome}.jpg")
        new_img_file_path = os.path.join(test_imgs_folder, f"{nome}.png")
        shutil.copy(img_file_path, new_img_file_path)


def write_txt_over_img(src_img, txt, center_x, center_y):
    """
    Write some text on the top left corner of an image
    and the circle center
    """

    # Definisci il punto in alto a sinistra (coordinate x, y)
    # in cui verrà posizionato il testo
    x = 10
    y = 30

    # Definisci il font, la grandezza del testo e il colore
    font = cv2.FONT_HERSHEY_SIMPLEX
    scala = 0.6
    colore = (0, 0, 255)

    # Scrivi il testo sull'immagine
    return_img = cv2.putText(
        src_img, txt, (x, y), font, scala, colore, thickness=2)

    # Circle center
    raggio = 3  # 5  2

    # Define the center of the rectangle
    center_x = int(center_x)
    center_y = int(center_y)

    # Draw Blue circle center
    return_img = cv2.circle(
        return_img, (center_x, center_y), raggio, colore, -1)
