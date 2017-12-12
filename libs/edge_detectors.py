"""
Edge detection library
INCLUDE: scikit-image
"""
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter
import numpy as np


class CheckingAngle(Enum):
    A = 0
    B = 1
    C = 2
    D = 3

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
    gauss_func = lambda x,y: -(x**2 + y**2 - 2*np.pi*(sigma**2))*np.exp([\
        -(x**2+y**2)/(2*np.pi*sigma**2)])\
        /((np.pi*(sigma**2)) ** 2)
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


def LOG(img, sigma, _kernel_size = 3):
    # calculo las varianzas locales en la imagen
    _out = np.zeros(img.shape)
    _N = len(img) # rows
    _M = len(img[0]) # cols
    # _laplacian = convolve2d(img, _LOG_kern(_kernel_size, sigma), mode='same')
    _laplacian = convolve2d(img, _gaussian_kern(_kernel_size, sigma), mode='same')
    _laplacian = laplacian(_laplacian)
    # print(_laplacian)
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
           # es borde, le asigno su valor en la imagen original
           _out[i,j] = img[i,j]
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


def kirsch_compass(img):
    def _rotate_kernel(kern):
        out = np.zeros(kern.shape)
        out[0,0] = kern[0,1]
        out[0,1] = kern[0,2]
        out[0,2] = kern[1,2]
        out[1,0] = kern[0,0]
        out[1,2] = kern[2,2]
        out[2,0] = kern[1,0]
        out[2,1] = kern[2,0]
        out[2,2] = kern[2,1]
        return out
    initial_kern = np.array([[5,5,5], [-3,0,-3], [-3,-3,-3]])
    directions = []
    for i in range(8):
        directions.append( convolve2d(img, initial_kern, mode='same') )
        initial_kern = _rotate_kernel(initial_kern)
    out = np.zeros(img.shape)
    for i in range(len(img)):
        for j in range(len(img[0])):
            out[i,j] = np.amax( [directions[d][i,j] for d in range(8)] )
    return out

# Sobel
def sobel_gradient(img):
    Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    J_x = convolve2d(img,Gx,mode='same')
    J_y = convolve2d(img,Gy,mode='same')
    J_norm = np.hypot(J_x, J_y)
    J_angle = np.arctan2(J_y, J_x)
    # J_norm = np.sqrt(np.power(J_x,2)+np.power(J_y,2))
    # J_angle = np.divide(J_y, J_x)
    return J_norm, J_angle


##############################################################################
### Canny Canny Canny Canny Canny Canny Canny Canny Canny Canny Canny Canny ##
##############################################################################

def canny(img, sigma, umin, umax):
    img = img.astype('int32') # para evitar overflow
    img = gaussian_filter(img, sigma)
    J_norm, J_angle = sobel_gradient(img)
    J_norm = non_maximum_supression(J_norm, J_angle)
    J_norm = umbral_por_histeresis(J_norm, umin, umax)
    return J_norm


def get_closest_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle

def non_maximum_supression(img, J_norm):
    res = np.zeros(img.shape, dtype=np.int32)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            direc = get_closest_angle(J_norm[i, j])
            if direc == 0:
                if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                    res[i,j] = img[i,j]
            elif direc == 90:
                if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                    res[i,j] = img[i,j]
            elif direc == 135:
                if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                    res[i,j] = img[i,j]
            elif direc == 45:
                if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                    res[i,j] = img[i,j]
            else:
                print('Falló el ángulo!')
                exit(1)
    return res

def umbral_por_histeresis(J_norm, umin, umax):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
    'WEAK': np.int32(50),
    'STRONG': np.int32(255),
    }

    # Clasifico a los pixels por weak/strong o los pongo en cero

    # get strong pixel indices
    strong_i, strong_j = np.where(J_norm > umax)
    # get weak pixel indices
    weak_i, weak_j = np.where((J_norm >= umin) & (J_norm <= umax))
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(J_norm < umin)
    # set values
    J_norm[strong_i, strong_j] = cf.get('STRONG')
    J_norm[weak_i, weak_j] = cf.get('WEAK')
    J_norm[zero_i, zero_j] = np.int32(0)

    weak = cf.get('WEAK')

    # Estiro los bordes según los umbrales
    strong = 255
    M, N = J_norm.shape
    for i in range(M):
        for j in range(N):
            if J_norm[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((J_norm[i + 1, j] == strong) or (J_norm[i - 1, j] == strong)
                         or (J_norm[i, j + 1] == strong) or (J_norm[i, j - 1] == strong)
                         or (J_norm[i+1, j + 1] == strong) or (J_norm[i-1, j - 1] == strong)):
                        J_norm[i, j] = strong
                    else:
                        J_norm[i, j] = 0
                except IndexError as e:
                    pass
    return J_norm
