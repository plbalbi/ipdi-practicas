"""
Edge detection library
INCLUDE: scikit-image
"""

import numpy as np
from scipy.signal import convolve2d

def _mean_kernel(d):
    return np.ones((d,d))*(1.0/float((d**2)))

def _gaussian_kern(dim, sigma):
    KERN = np.zeros((dim, dim))
    gauss_func = lambda x,y: (1/(2*np.pi*sigma))*np.exp([\
        -(x**2 + y**2)/(2*sigma**2)\
        ])
    for i in range(dim):
        for j in range(dim):
            KERN[i,j] = gauss_func(abs(i-dim/2),abs(j-dim/2))
    return KERN


def _LOG_kern(dim, sigma):
    KERN = np.zeros((dim, dim))
    gauss_func = lambda x,y: (x**2 + y**2 - 2*np.pi*sigma**2)*np.exp([\
        -(x**2+y**2)/(2*np.pi*sigma**2)])\
        /((np.pi*sigma**2) ** 2)
    for i in range(dim):
        for j in range(dim):
            KERN[i,j] = gauss_func(abs(i-dim/2),abs(j-dim/2))
    return KERN

def laplacian(img):
    laplacian_kernel = np.array([[0,1,0],\
                                 [1,-4,1],\
                                 [0,1,0]])
    return convolve2d(img, laplacian_kernel, mode='same')

# Laplacian + Local variance threshold
def LLV(img, th, _kernel_size = 3, _smooth_pre_laplacian = False):
    # calculo las varianzas locales en la imagen
    _out = np.zeros(img.shape)
    _means = convolve2d(img, _mean_kernel(_kernel_size), mode='same')
    _diff_means_2 = np.power(img - _means, 2)
    _local_variance = convolve2d(_diff_means_2, _mean_kernel(_kernel_size), mode='same') 
    _N = len(img) # rows
    _M = len(img[0]) # cols
    if _smooth_pre_laplacian:
        img_smoothed = convolve2d(img, _mean_kernel(3), mode='same') 
        _laplacian = laplacian(img_smoothed)
    else:
        _laplacian = laplacian(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
           # check zero point crossing 
           _is_zero_cross = False
           # im in corner
           if (i,j) in [(0,0), (_N-1, 0), (0,_M-1), (_N-1,_M-1)]:
               if (i,j) == (0,0):
                   _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [6,5,4])
               elif (i,j) == (_N-1, 0):
                   _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [2,3,4])
               elif (i,j) == (0, _M-1):
                   _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [8,7,6])
               else:
                   _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [1,8,2])
           # im in left_border
           elif j == 0: 
               _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [2,3,4,5,6])
           # im in right_border
           elif j == _M-1:
               _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [1,2,8,7,6])
           # im in top_border
           elif i == 0:
               _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [8,4,7,6,5])
           # im in bottom_border
           elif i == _N-1:
               _is_zero_cross = _check_zero_crossing(_laplacian, i, j, [1,2,3,8,4])
           else: 
               _is_zero_cross = _check_zero_crossing(_laplacian, i, j, range(1,9))
           if not _is_zero_cross:
               # no es borde
               continue
           #check local variance threshold
           if _local_variance[i,j] > th:
               # es borde, le asigno su valor en la imagen original
               _out[i,j] = img[i,j]
           # no es borde
    return _out

def _check_zero_crossing(img,i,j,positions, _zero_th = 10):
    """
    Checkea en un vecinadrio de 3x3 al rededor de donde estoy,
    si en alguna direccion se cruza por el cero, es decir, si
    un valor es positivo y otro negativo.
    1 2 3
    8 X 4
    7 6 5
    """
    if 1 in positions and\
       ((img[i,j] * img[i-1,j-1] < 0) and\
        abs(img[i,j] - img[i-1,j-1]) > _zero_th):
               return True
    if 2 in positions and\
       ((img[i,j] >= 0 * img[i-1,j] < 0) and\
        abs(img[i,j] - img[i-1,j]) > _zero_th):
               return True
    if 3 in positions and\
       ((img[i,j] * img[i-1,j+1] < 0) and\
        abs(img[i,j] - img[i-1,j+1]) > _zero_th):
               return True
    if 4 in positions and\
       ((img[i,j] * img[i,j+1] < 0) and\
        abs(img[i,j] - img[i,j+1]) > _zero_th):
               return True
    if 5 in positions and\
       ((img[i,j] * img[i+1,j+1] < 0) and\
        abs(img[i,j] - img[i+1,j+1]) > _zero_th):
               return True
    if 6 in positions and\
       ((img[i,j] * img[i+1,j] < 0) and\
        abs(img[i,j] - img[i+1,j]) > _zero_th):
               return True
    if 7 in positions and\
       ((img[i,j] * img[i+1,j-1] < 0) and\
        abs(img[i,j] - img[i+1,j-1]) > _zero_th):
               return True
    if 8 in positions and\
       ((img[i,j] * img[i,j-1] < 0) and\
        abs(img[i,j] - img[i,j-1]) > _zero_th):
               return True
    return False


