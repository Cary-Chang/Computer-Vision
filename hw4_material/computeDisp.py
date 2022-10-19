from random import random
import numpy as np
import cv2.ximgproc as xip
# import cv2
# from numpy import matlib

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    Il_pad = np.pad(Il, 1, 'edge')[:, :, 1:-1]
    Ir_pad = np.pad(Ir, 1, 'edge')[:, :, 1:-1]
    BP_l = None
    BP_r = None
    cost_l = np.zeros((h, w, max_disp+1), dtype=np.float32)
    cost_r = np.zeros((h, w, max_disp+1), dtype=np.float32)

    for y in range(3):
        for x in range(3):
            if y != 1 and x != 1:
                if BP_l is not None:
                    BP_l = np.dstack((BP_l, Il_pad[y:y+h, x:x+w] <= Il))
                    BP_r = np.dstack((BP_r, Ir_pad[y:y+h, x:x+w] <= Ir))
                else:
                    BP_l = (Il_pad[y:y+h, x:x+w] <= Il)
                    BP_r = (Ir_pad[y:y+h, x:x+w] <= Ir)

    for disp in range(max_disp + 1):
        cost_overlap = np.sum(BP_l[:, disp:w] ^ BP_r[:, :w-disp], axis=2)
        cost_l[:, disp:, disp] = cost_overlap
        cost_l[:, :disp, disp] = cost_overlap[:, [0]]
        cost_r[:, :w-disp, disp] = cost_overlap
        cost_r[:, w-disp:, disp] = cost_overlap[:, [-1]]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for disp in range(max_disp + 1):
        cost_l[:, :, disp] = xip.jointBilateralFilter(Il, cost_l[:, :, disp], -1, 6, 6)
        cost_r[:, :, disp] = xip.jointBilateralFilter(Ir, cost_r[:, :, disp], -1, 6, 6)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    disp_l = np.argmin(cost_l, axis=2)
    disp_r = np.argmin(cost_r, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    for y in range(h):
        for x in range(w):
            if x - disp_l[y, x] >= 0 and disp_l[y, x] == disp_r[y, x - disp_l[y, x]]:
                continue
            disp_l[y, x] = -1

    for y in range(h):
        for x in range(w):
            if disp_l[y, x] == -1:
                FL = FR = max_disp
                if x != 0:
                    left_pt = disp_l[y, :x]
                    left_pt_idx = np.where(left_pt >= 0)
                    if left_pt_idx[0].size != 0:
                        FL = left_pt[left_pt_idx[0][-1]]

                if x != w - 1:
                    right_pt = disp_l[y, x+1:]
                    right_pt_idx = np.where(right_pt >= 0)
                    if right_pt_idx[0].size != 0:
                        FR = right_pt[right_pt_idx[0][0]]

                disp_l[y, x] = min(FL, FR)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), disp_l.astype(np.uint8), 12, 3)
    return labels.astype(np.uint8)
    