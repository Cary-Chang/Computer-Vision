import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None
    
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux = np.reshape(u[:,0], (-1, 1))
    uy = np.reshape(u[:,1], (-1, 1))
    vx = np.reshape(v[:,0], (-1, 1))
    vy = np.reshape(v[:,1], (-1, 1))
    A1 = np.hstack((u, np.ones((N, 1)), np.zeros((N, 3)), -(ux * vx), -(uy * vx), -vx))
    A2 = np.hstack((np.zeros((N, 3)), u, np.ones((N, 1)), -(ux * vy), -(uy * vy), -vy))    
    A = np.vstack((A1, A2))
    # TODO: 2.solve H with A
    _, _, vh = np.linalg.svd(A)
    h = vh[-1,:]
    h /= vh[-1,-1]
    H = np.reshape(h, (3, 3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    XX = np.vstack((np.reshape(xx, (1, -1)), np.reshape(yy, (1, -1)), np.ones((1, xx.size))))
    if direction == 'b':
        vx, vy, X_prime = xx, yy, XX
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        X = H_inv @ X_prime
        X /= X[-1,:]
        ux = np.reshape(X[0,:], (ymax - ymin, xmax - xmin))
        uy = np.reshape(X[1,:], (ymax - ymin, xmax - xmin))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.where((0 < ux) & (ux < w_src - 1) & (0 < uy) & (uy < h_src - 1))
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        ux, uy = np.round(ux[mask]).astype(int), np.round(uy[mask]).astype(int)
        # TODO: 6. assign to destination image with proper masking
        dst[vy[mask], vx[mask], :] = src[uy, ux, :]

    elif direction == 'f':
        ux, uy, X = xx, yy, XX
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        X_prime = H @ X
        X_prime /= X_prime[-1,:]
        vx = np.reshape(X_prime[0,:], (ymax - ymin, xmax - xmin))
        vy = np.reshape(X_prime[1,:], (ymax - ymin, xmax - xmin))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.where((0 < vx) & (vx < w_dst - 1) & (0 < vy) & (vy < h_dst - 1))
        # TODO: 5.filter the valid coordinates using previous obtained mask
        vx, vy = np.round(vx[mask]).astype(int), np.round(vy[mask]).astype(int)
        # TODO: 6. assign to destination image using advanced array indicing
        dst[vy, vx, :] = src[uy[mask], ux[mask], :]

    return dst
