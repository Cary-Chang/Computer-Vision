from turtle import width
import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        gaussian_images.append(image)
        for i in range(1, self.num_guassian_images_per_octave):
            img = cv2.GaussianBlur(image, (0, 0), self.sigma ** i)
            gaussian_images.append(img)
        width = image.shape[1]
        height = image.shape[0]
        scaled_image = cv2.resize(gaussian_images[-1], (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_NEAREST)
        new_width = scaled_image.shape[1]
        new_height = scaled_image.shape[0]
        gaussian_images.append(scaled_image)
        for i in range(1, self.num_guassian_images_per_octave):
            img = cv2.GaussianBlur(scaled_image, (0, 0), self.sigma ** i)
            gaussian_images.append(img)
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(2 * self.num_DoG_images_per_octave + 1):
            if i != self.num_DoG_images_per_octave:
                img = cv2.subtract(gaussian_images[i], gaussian_images[i+1])
                dog_images.append(img)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_DoG_images_per_octave - 2):
            for j in range(1, height - 1):
                for k in range(1, width - 1):
                    if abs(dog_images[i+1][j, k]) > self.threshold:
                        sub_img = dog_images[i+1][j-1:j+2, k-1:k+2]
                        pre_sub_img = dog_images[i][j-1:j+2, k-1:k+2]
                        next_sub_img = dog_images[i+2][j-1:j+2, k-1:k+2]
                        cube = np.dstack((pre_sub_img, sub_img, next_sub_img))
                        # cube = np.transpose(cube, (2, 0, 1))
                        if dog_images[i+1][j, k] == np.max(cube) or dog_images[i+1][j, k] == np.min(cube):
                            keypoints.append([j, k])
        
        for i in range(self.num_DoG_images_per_octave, 2 * self.num_DoG_images_per_octave - 2):
            for j in range(1, new_height - 1):
                for k in range(1, new_width - 1):
                    if abs(dog_images[i+1][j, k]) > self.threshold:
                        sub_img = dog_images[i+1][j-1:j+2, k-1:k+2]
                        pre_sub_img = dog_images[i][j-1:j+2, k-1:k+2]
                        next_sub_img = dog_images[i+2][j-1:j+2, k-1:k+2]
                        cube = np.dstack((pre_sub_img, sub_img, next_sub_img))
                        # cube = np.transpose(cube, (2, 0, 1))
                        if dog_images[i+1][j, k] == np.max(cube) or dog_images[i+1][j, k] == np.min(cube):
                            keypoints.append([j * 2, k * 2])
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.array(keypoints)
        keypoints = np.unique(keypoints, axis=0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
