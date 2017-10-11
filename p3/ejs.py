import fourier
import math
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.fftpack import ifft2, fft2
from scipy import misc
import sys

def ej1a():
    N = 8
    xx = np.arange(0.0, N, 0.1)
    xx_d = range(N)

    subplot = 1
    for k in range(N):
        yy = np.real([np.exp(2j*math.pi*x*k/N) for x in xx])
        yy_d = np.real([np.exp(2j*math.pi*x*k/N) for x in xx_d])
        plt.subplot(8,1,subplot)
        plt.plot(xx,yy)
        plt.plot(xx_d,yy_d,'ro')
        subplot+=1
    plt.show()


def ej1b():
    N = 8
    xx = np.arange(0.0, N, 0.1)
    xx_d = range(N)

    subplot = 1
    for m in range(N):
        for n in range(N):
            yyd = [np.real([np.exp(2j*math.pi*(m*k+n*l)/N) for k in xx_d]) for l in xx_d]
            plt.subplot(8,8,subplot)
            plt.axis('off')
            plt.imshow(yyd,cmap='gray')
            subplot += 1

    plt.show()


def ej2():
    N = 15
    x = []
    for i in range(N):
        x.append(random.randint(0,5))
    F_x = fourier.DFT(x)
    altos = range(int(2/3*N),N)
    medios = range(int(1/3*N),int(2/3*N))
    bajos = range(int(N/3))
    # suprimo frecuencias altas
    F_x_h = F_x[:]
    for i in altos:
        F_x_h[i] = 0
        x_h = fourier.IDFT(F_x_h)
    # suprimo frecuencias bajas
    F_x_l = F_x[:]
    for i in bajos:
        F_x_l[i] = 0
        x_l = fourier.IDFT(F_x_l)
    # suprimo frecuencias medias
    F_x_m = F_x[:]
    for i in medios:
        F_x_m[i] = 0
        x_m = fourier.IDFT(F_x_m)

    plt.plot(x,label='Señal original')
    plt.plot(x_h,label='Sin frecuencias altas')
    plt.plot(x_l,label='Sin frecuencias bajas')
    plt.plot(x_m,label='Sin frecuencias medias')
    plt.legend()
    plt.show()

def ej3():

    a = [[1,2],[3,4]]
    F_a = fourier.DFT_2D(a)
    a_i = fourier.IDFT_2D(F_a)
    print(a)
    print('\n')
    print(F_a)
    print('\n')
    print(a_i)

def complex_norm(X):
    return np.sqrt(np.add( np.square(np.real(X)), np.square(np.imag(X))))

def ej4():
    if len(sys.argv) != 3:
        print("faltan params")
        sys.exit(1)
    # fft2
    im1 = misc.imread(sys.argv[1])
    # im2 = imread(sys.argv[2])
    im1_FT = fft2(im1)
    print(im1_FT)
    # im2_FT = fft2(im2)
    
    # te da el angulo del complex
    # plt.imshow(np.angle(im1_FT))
    im1_norms = complex_norm(im1_FT)
    print(im1_norms)
    plt.imshow(im1_norms, cmap='gray', vmin=0, vmax=np.amax(im1_norms))
    plt.show()

    temp = ifft2(im1_FT)
    plt.imshow(np.uint8(np.real(temp)), cmap='gray')
    plt.show()

    # print(np.uint8(np.real(temp)))


ej4()
