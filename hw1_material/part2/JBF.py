import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        x, y = np.meshgrid(np.arange(2 * self.pad_w + 1) - self.pad_w, np.arange(2 * self.pad_w + 1) - self.pad_w)
        Gs = np.exp(-(x * x + y * y) * 0.5 / (self.sigma_s * self.sigma_s))
        Gr = np.exp(-np.arange(256) * np.arange(256) * 0.5 / (self.sigma_r * self.sigma_r * 255 * 255))
        output = np.zeros(img.shape).astype(np.float64)
        w = np.zeros(img.shape).astype(np.float64)
        width = img.shape[1]
        height = img.shape[0]
        if guidance.ndim == 2:
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    h = Gs[i, j] * Gr[abs(guidance - padded_guidance[i:i+height, j:j+width])]
                    h = np.dstack((h, h, h))
                    w += h
                    output += h * padded_img[i:i+height, j:j+width, :]
        else:
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    h = Gs[i, j] * Gr[abs(guidance[:, :, 0] - padded_guidance[i:i+height, j:j+width, 0])] \
                                 * Gr[abs(guidance[:, :, 1] - padded_guidance[i:i+height, j:j+width, 1])] \
                                 * Gr[abs(guidance[:, :, 2] - padded_guidance[i:i+height, j:j+width, 2])] 
                    h = np.dstack((h, h, h))
                    w += h
                    output += h * padded_img[i:i+height, j:j+width, :]
        output /= w
        return np.clip(output, 0, 255).astype(np.uint8)