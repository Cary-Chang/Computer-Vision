import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        accepted_U = []
        accepted_V = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                accepted_V.append(kp1[m.queryIdx].pt)
                accepted_U.append(kp2[n.trainIdx].pt)
        X = np.array(accepted_U, dtype=int)
        X = np.vstack((X.T, np.ones((1, X.shape[0]))))
        X_prime = np.array(accepted_V, dtype=int)
        X_prime = np.vstack((X_prime.T, np.ones((1, X_prime.shape[0]))))
        # TODO: 2. apply RANSAC to choose best H
        iters = 36000
        threshold = 3.7
        sub_samples = 8
        max_inlier = 0
        best_H = None
        for _ in tqdm(range(iters)):
            sub_samples_idx = random.sample(range(X.shape[1]), sub_samples)
            H = solve_homography(X[:-1, sub_samples_idx].T, X_prime[:-1, sub_samples_idx].T)
            X_bar = H @ X
            if (X_bar[-1,:] <= 1e-8).any():
                continue
            X_bar /= X_bar[-1,:]
            error = np.linalg.norm(X_bar - X_prime, axis=0)
            num_inlier = np.sum(error < threshold)
            if num_inlier > max_inlier:
                max_inlier = num_inlier
                best_H = H
        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, dst.shape[0], 0, dst.shape[1], 'b')        
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)