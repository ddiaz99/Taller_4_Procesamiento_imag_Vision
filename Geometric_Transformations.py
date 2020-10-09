#Se importan las librerias
import numpy as np
import os
import cv2

def click_event(event, x, y, flags, params): #funcion para atender el evento del mouse
    global number_points #variable que nos permite contar cuantos puntos se han seleccionado
    if number_points < 3:
        if event == cv2.EVENT_RBUTTONDOWN:
            mouse_coordinates.append([x, y]) #se guardan en la variable mouse_coordinates el punto seleccionado
            number_points += 1
    else: #si se seleccionaron 3 puntos se cierra la ventana
        number_points = 0
        cv2.destroyAllWindows()

def get_points(): #funcion para guardar los puntos
    for i in range(0,len(images)): #ciclo para guardar 6 puntos, 3 por cada imagen
        cv2.imshow(windowsNames[i],images[i]) #se muestra la imagen
        cv2.setMouseCallback(windowsNames[i], click_event) #se llama a la funcion para seleccionar puntos
        cv2.waitKey(0)

    pts1 = np.float32(mouse_coordinates[0:3]) #3 puntos seleccionados para la imagen lena
    pts2 = np.float32(mouse_coordinates[3:6]) #3 puntos seleccionados para la imagen lena_warped

    return pts1,pts2

def estimate_similarity(pts1,M_affine): #funcion para estimar la transformacion de similitud (metodo explicado en el documento)

    #se transforman los puntos de la primera imagen a coordenadas homogeneas
    point1_homogeneous = np.append(pts1[0], np.array([1]), axis=0)
    point2_homogeneous = np.append(pts1[1], np.array([1]), axis=0)
    point3_homogeneous = np.append(pts1[2], np.array([1]), axis=0)

    #se realiza la transformacion de estos puntos a partir de la transformacion afin
    new_point1 = M_affine.dot(point1_homogeneous)
    new_point2 = M_affine.dot(point2_homogeneous)
    new_point3 = M_affine.dot(point3_homogeneous)

    x1 = point1_homogeneous[0]
    y1 = point1_homogeneous[1]
    x2 = point2_homogeneous[0]
    y2 = point2_homogeneous[1]
    x3 = point3_homogeneous[0]
    y3 = point3_homogeneous[1]


    x1_prime = new_point1[0]
    y1_prime = new_point1[1]
    x2_prime = new_point2[0]
    y2_prime = new_point2[1]
    x3_prime = new_point3[0]
    y3_prime = new_point3[1]

    #vector solucion con los puntos transformados
    b = np.array([x1_prime,y1_prime,x2_prime,y2_prime,x3_prime,y3_prime])

    #se crea la matriz correspondiente con los 3 puntos
    matrix1 = np.array([[x1,y1,1,0,0,0],[0,0,0,x1,y1,1],[x2,y2,1,0,0,0],[0,0,0,x2,y2,1],[x3,y3,1,0,0,0],[0,0,0,x3,y3,1]])
    matrix2 = matrix1.T #se traspone la matriz

    A_1 = np.linalg.inv(matrix2.dot(matrix1)) #se calcula la inversa de ese producto
    A_2 = A_1.dot(matrix2) #se multiplica la matriz anterior por la matriz traspuesta
    A_3 = np.dot(A_2,b.T) #por ultimo se multiplica por el vector solucion

    #parametros de la matriz de similitud
    a = A_3[0]
    b = A_3[1]
    tx = A_3[2]
    d = A_3[3]
    e = A_3[4]
    ty = A_3[5]

    #computacion de los parametros theta, (s1,s2)
    theta = np.arctan(-b/e)
    s1 = a/np.cos(theta)
    s2 = e/np.cos(theta)

    print('approximation of similarity matrix parameters')
    print(f'theta1 = {theta}')
    print(f's1 = {s1}')
    print(f's2 = {s2}')
    print(f'tx = {tx}')
    print(f'ty = {ty}')

    #matriz de similitud obtenida
    M_similarity = np.array([[a, b, tx],
                        [d, e, ty]])

    return M_similarity

def calc_error(pts1,M_similarity,pts2): #funcion para calcular el error

    # se le agregar una fila de unos (coordenadas homogeneas) a la traspuesta de los primeros puntos
    # esto con el fin de poder multiplicar por la matriz de similitud obtenida
    pts = np.append(pts1.transpose(), np.array([[1, 1, 1]]), axis=0)
    pts_transform = M_similarity.dot(pts)

    # se vuelve a trasponer la matriz para obtener los puntos originales ahora transformados
    pts_transform = pts_transform.transpose()

    # se calcula la norma de la diferencia de cada punto (arreglo de 3 con el error en cada caso)
    error = np.linalg.norm(pts_transform - pts2, axis=1)
    print('Error calculated = ',error)

if __name__ == '__main__':
    mouse_coordinates = list() #lista donde se guardan las coordenadas escogidas con el mouse
    number_points = 0 #variable que nos permite contar cuantos puntos se han seleccionado, se inicializa en cero

    #r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'
    path1 = input('ingrese path: ') #se pide al usuario que ingrese el path donde se encuentran las imagenes
    image_name1 = input('image name: ') #r'lena.png' se pide al usuario que ingrese el nombre de la imagen
    path_file1 = os.path.join(path1, image_name1)
    lena = cv2.imread(path_file1, 1) #se lee la imagen segun la ruta

    image_name2 = input('image name: ') #r'lena_warped.png' se pide al usuario que ingrese el nombre de la imagen
    path_file2 = os.path.join(path1, image_name2)
    lena_warped = cv2.imread(path_file2, 1)

    windowsNames = ['lena','lena_warped'] #lista con los nombres de las imagenes
    images = [lena,lena_warped] #lista con las dos imagenes seleccionadas

    pts1,pts2 = get_points() #se obtienen los 6 puntos de las dos imagenes

    M_affine = cv2.getAffineTransform(pts1, pts2) #se guarda en M_affine la matriz de transformacion afin

    print('Affine transformation: ')
    print(M_affine)

    #se realiza la transformacion a la imagen lena
    image_affine = cv2.warpAffine(lena, M_affine, lena.shape[:2])
    cv2.imshow("Affine", image_affine)

    #se realiza la estimacion de la transformacion de similitud y
    M_similarity = estimate_similarity(pts1,M_affine) #matriz de similitud

    #se realiza la tranformacion de similitud a la imagen lena
    image_similarity = cv2.warpAffine(lena, M_similarity, lena.shape[:2])
    cv2.imshow('similarity', image_similarity)

    #se calcula el error
    calc_error(pts1, M_similarity, pts2)

    cv2.waitKey(0)