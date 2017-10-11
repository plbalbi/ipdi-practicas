import fourier
import math
from matplotlib import pyplot as plt
import numpy as np
import random

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
