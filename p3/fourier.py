import numpy as np
# convolucion
def convolve(I, m):
# def convolve(I, m, z_padding = True):
    # I es la imagen
    image_width = I.shape[1]
    image_heigth = I.shape[0]
    # m es la máscara, debe ser de m x n
    # size de la máscara
    kern_m = m.shape[0] 
    kern_n = m.shape[1] 
    # kernel center
    kern_m_center = int((kern_m-1)/2)
    kern_n_center = int((kern_n-1)/2)
    # generate arrays to index
    kern_m_indexes = [a-kern_m_center for a in range(0, kern_m)]
    # kern_m_indexes = range(0, kern_m)
    # kern_m_indexes = map(lambda x: x - kern_m_center,kern_m_indexes)


    kern_n_indexes = [a-kern_n_center for a in range(0, kern_n)]
    # kern_n_indexes = range(0, kern_n)
    # kern_n_indexes = map(lambda x: x - kern_n_center,kern_n_indexes)


    res_image = np.zeros(I.shape)

    for i in range(0, I.shape[0]):
        for j in range(0, I.shape[1]):
            # pixel I(i,j)
            local_sum = 0
            for row_index in kern_m_indexes:
                # if i+row_index in range(kern_m_indexes[0],kern_m_center):
                    # #TODO: doy con borde sup
                    # IM = I[i+row_index,range((j+kern_n_indexes[0]),(j+kern_n_indexes[-1])+1)]
                    # continue
                # elif i+row_index in \
                        # range(image_heigth-kern_m_center,image_heigth+kern_m_indexes[-1]):
                    # #TODO: doy con borde inf
                    # continue
                if i+row_index < 0 or i+row_index > image_heigth-1:
                    continue
                else:
                    # sin borde arriba o abajo, en cuyo caso se anula, ya que
                    # lo estoy tratando por filas
                    # me armo un vector por finla que se multiplica, y uso prod interno
                    KERN = m[kern_m_center+row_index, :]
                    if j in range(0,kern_n_center):
                        #TODO: doy con borde izq
                        # IM = [0]*cantidad que se va de rango + pedazo de la imagen que no se va
                        IM = np.append([0]*(kern_n_center-j), I[i+row_index,range(0,kern_n_indexes[-1]-j+1)])
                    elif j in range(image_width-1-kern_n_center+1, image_width): 
                        #[WIDTH - center , WIDTH -1]
                        #TODO: doy con borde der
                        # IM = pedazo de la imagen que no se va + [0]*cantidad que se va de rango
                        # image_width-1-j == 0 si j == image_width-1
                        # image_width-1-j == pixels que faltan hasta ultima posicion
                        IM = np.append(I[i+row_index,range(j+kern_n_indexes[0], image_width)], [0]*(kern_n_center-(image_width-1-j))) 
                    else:
                        # no doy con borde a los costados
                        IM = I[i+row_index,range((j+kern_n_indexes[0]),(j+kern_n_indexes[-1])+1)]
                        # print("Accediendo a %d, [%d, %d]" % (i+row_index,\
                                # (j+kern_n_indexes[0]),(j+kern_n_indexes[-1])))
                        # print(IM)
                    try:
                        local_sum+= np.inner(IM, KERN)
                    except:
                        print("i=%d | j=%d | row_index= %d" % (i,j,row_index))
                        print("IM = ", IM)
                        print("KERN = ", KERN)
                        sys.exit(1)
            res_image[i,j] = local_sum
    return res_image


def fourier_transf():
    return
    # transformada
