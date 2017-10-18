import sys
sys.path.insert(0, '../libs')
from fourier import *
import math
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.fftpack import ifft2, fft2
from scipy import misc
from scipy import fftpack
from scipy import signal
from scipy import ndimage
import sys
import ecualizacion as equ

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
    plt.savefig('informe/imgs/1a.pdf')
    plt.show()


def ej1b():
    N = 8
    xx = np.arange(0.0, N, 0.05)
    xx_d = range(N)

    subplot = 1
    for n in range(N):
        for m in range(N):
            yyd = [np.real([np.exp(2j*math.pi*(m*k+n*l)/N) for k in xx_d]) for l in xx_d]
            plt.subplot(8,8,subplot)
            plt.axis('off')
            plt.imshow(yyd,cmap='gray')
            subplot += 1

    plt.savefig('informe/imgs/1b.pdf')


def ej2():
    N = 10
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
    plt.savefig('informe/imgs/2.pdf')
    plt.show()

def ej3():
    # Creando las imágenes para cada ejercicio...
    N = 100
    img_a = [[0 for i in range(N)] for j in range(N)]
    img_b = [[0 for i in range(N)] for j in range(N)]
    img_c = [[0 for i in range(N)] for j in range(N)]
    for i in range(int(N/2)-15,int(N/2)+15):
        for j in range(int(N/2)-15,int(N/2)+15):
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

    imgs = [img_a,img_b,img_c,img_d,img_e,img_f,img_g,img_h,img_i,img_j]

    it = 0
    for img in imgs:
        plt.figure(figsize=(10,4))
        plt.subplot(1,4,1)
        plt.title('Original')
        plt.axis('off')
        plt.imshow(img,cmap='gray')
        plt.subplot(1,4,2)
        plt.axis('off')
        plt.title('DFT (absoluto)')
        plt.imshow(np.abs(fftpack.fft(img)),cmap='gray')
        plt.subplot(1,4,3)
        plt.axis('off')
        plt.title('DFT (real)')
        plt.imshow(np.real(fftpack.fft(img)),cmap='gray')
        plt.subplot(1,4,4)
        plt.axis('off')
        plt.title('DFT (ángulo)')
        plt.imshow(np.angle(fftpack.fft(img)),cmap='gray')
        plt.suptitle('Imagen '+chr(97+it))
        plt.tight_layout()
        plt.savefig('informe/imgs/3-'+chr(97+it))
        it += 1
        # plt.show()

def ej4():
    # if len(sys.argv) != 3:
    #     print("faltan params")
    #     print("llamar como >"+ '\033[1m' +" %s imagen1 imagen2" % (sys.argv[0])+ '\033[0m')
    #     sys.exit(1)
    # fft2
    # im1 = misc.imread(sys.argv[1])
    # im2 = misc.imread(sys.argv[2])
    im1 = misc.imread('lena.png')
    im2 = misc.imread('ladrillos.png')
    im1_FT = fft2(im1)
    im2_FT = fft2(im2)
    im1_norm, im1_angle = uncompress_cmpx(im1_FT)
    im2_norm, im2_angle = uncompress_cmpx(im2_FT)

    reconstruct_1 = ifft2(\
            assemble_complex(im1_norm, im2_angle)\
            )

    plt.subplot(1,3,1)
    plt.title("tomo la norma de aca")
    plt.imshow(equ_feo_fft(im1_norm), cmap='gray')

    plt.subplot(1,3,2)
    plt.title("tomo phase angle de aca")
    plt.imshow(im2_angle, cmap='gray')

    plt.subplot(1,3,3)
    plt.title("resultado de la comoposicion de ambas")
    plt.imshow(IFFT_TO_UINT8(reconstruct_1), cmap='gray')

    plt.show()

    misc.imsave("informe/imgs/ej4_A.PNG", IFFT_TO_UINT8(reconstruct_1))

    reconstruct_2 = ifft2(\
            assemble_complex(im2_norm, im1_angle)\
            )

    plt.subplot(1,3,1)
    plt.title("tomo la norma de aca")
    plt.imshow(equ_feo_fft(im2_norm), cmap='gray')

    plt.subplot(1,3,2)
    plt.title("tomo phase angle de aca")
    plt.imshow(im1_angle, cmap='gray')

    plt.subplot(1,3,3)
    plt.title("resultado de la comoposicion de ambas")
    plt.imshow(IFFT_TO_UINT8(reconstruct_2), cmap='gray')

    plt.show()

    misc.imsave("informe/imgs/ej4_B.PNG", IFFT_TO_UINT8(reconstruct_2))

    # te da el angulo del complex (radianes cualculo)
    # np.angle(im1_FT)
    # im1_norms = complex_norm(im1_FT)
    # print(im1_norms)
    # plt.imshow(im1_norms, cmap='gray', vmin=0, vmax=np.amax(im1_norms))
    # plt.show()

    # temp = ifft2(im1_FT)
    # plt.imshow(np.uint8(np.real(temp)), cmap='gray')
    # plt.show()

    # print(np.uint8(np.real(temp)))

def ej5():
    lena_route = "lena.png"
    lena_img = misc.imread(lena_route)
    lena_FFT = fft2(lena_img)
    # lineas horizontales

    lena_FFT[50][0] += 1.5e6

    img = IFFT_TO_UINT8(ifft2(lena_FFT))

    plt.imshow(img, cmap='gray')
    plt.show()

def ej6():
    f = [random.randint(0,5) for i in range(10)]
    g = [random.randint(0,5) for i in range(10)]
    fg = ndimage.convolve(g,f,mode='wrap')
    F1 = fftpack.fft(fg)

    F2 = np.multiply(fftpack.fft(f),fftpack.fft(g))

    plt.subplot(2,2,1)
    plt.plot(F1,'go',label='F(f*g)',alpha=0.5)
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(F2,'ro',label='F(f).F(g)',alpha=0.5)
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(f,'go',label='F(f*g)',alpha=0.5)
    plt.subplot(2,2,4)
    plt.plot(g,'go',label='F(f*g)',alpha=0.5)
    plt.show()


def test_norms():
    # con 2 de gamma se ve bien
    img = misc.imread('lena.png')
    gamma = 2
    img_FFT = fft2(img)
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,3,2)
    # plt.imshow(fix_norm_plot_regions(equ_feo_fft(np.abs(img_FFT))),cmap='gray')
    plt.imshow(fix_norm_plot_regions(log_transform(np.abs(img_FFT), gamma)),cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(fix_norm_plot_regions(np.angle(img_FFT)),cmap='gray')
    plt.show()

test_norms()
