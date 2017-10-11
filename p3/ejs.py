import fourier
import math
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy import misc

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
    # Creando las imágenes para cada ejercicio...
    N = 100
    img_a = [[0 for i in range(N)] for j in range(N)]
    img_b = [[0 for i in range(N)] for j in range(N)]
    img_c = [[0 for i in range(N)] for j in range(N)]
    for i in range(int(N/2)-15,int(N/2)+15):
        for j in range(int(N/2)-15,int(N/2)+15):
            print(i,j)
            img_a[i][j] = 1
            img_b[i+20][j+20] = 1
            img_c[i][j] = 1
            img_c[i+1][j] = 1
            img_c[i+2][j] = 1
            img_c[i+3][j] = 1
            img_c[i+4][j] = 1
            img_c[i+5][j] = 1
            img_c[i+6][j] = 1
            img_c[i+7][j] = 1
            img_c[i+8][j] = 1
            img_c[i+9][j] = 1
            img_c[i+10][j] = 1
            img_c[i+11][j] = 1
    img_d = [[0 for i in range(N)] for j in range(N)]
    for i in range(10,int(N/3)):
        for j in range(int(N/2)-20,int(N/2)+30):
            img_d[i][j] = 1
    for i in range(80,95):
        for j in range(15,20):
            img_d[i][j] = 1
    img_e = [[0 for i in range(N)] for j in range(N)]
    img_h = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        img_e[i][45] = 1
        img_h[i][23] = 1
        img_h[i][65] = 1
        img_h[i][95] = 1
    img_f = misc.imrotate(img_e,45)
    img_g = misc.imrotate(img_e,90)
    img_i = misc.imrotate(img_h,45)
    img_j = misc.imrotate(img_h,90)
    # fin...




ej3()
