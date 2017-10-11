import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
import fourier as fourier
import sys


test_img = misc.imread(sys.argv[1])
plt.subplot(1,2,1)
plt.imshow(test_img, cmap='gray')
plt.subplot(1,2,2)

M_KERN = np.array([\
        [-1, 0, 1],\
        [-2, 0, 2],\
        [-1, 0, 1]\
        ], dtype = np.int)

M_KERN_AVG = (1/9)*np.ones([3,3])
print(M_KERN)
# import pdb; pdb.set_trace()
conv_img = fourier.convolve(test_img, M_KERN)
plt.imshow(conv_img, cmap='gray')
plt.show()
