import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
from matplotlib import pyplot as plt


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    DoG = Difference_of_Gaussian(args.threshold)
    # _, dog_images = DoG.get_keypoints(img)
    # for i, img in enumerate(dog_images):
    #     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #     plt.figure(i)
    #     plt.imshow(img, cmap='gray')
    # plt.show()

    keypoints = DoG.get_keypoints(img)
    rgb_img = cv2.imread(args.image_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    for p in keypoints:
        cv2.circle(rgb_img, (p[1], p[0]), 2, (0, 255, 0), -1)
    plt.figure(1)
    plt.imshow(rgb_img)
    plt.show()

if __name__ == '__main__':
    main()