#Se importan las librerias
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def recreate_image(centers, labels, rows, cols): #funcion que permite recrear la imagen segun el numero de clusters
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1

    plt.figure(1)
    plt.clf()
    plt.axis('off')
    plt.title('Quantized image ({} colors, method={})'.format(n_colors, metodo))
    plt.imshow(image_clusters)

    plt.show()

def clustering(image,metodo): #funcion para crear modelos segun el metodo a utilizar

    # labels guarda las predicciones de color para cada pixel de la imagen
    # centers guarda los valores de los centros obtenidos
    if metodo == 'kmeans':
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample) #se crea un modelo KMEANS segun n_colors
        labels = model.predict(image_array)
        centers = model.cluster_centers_

    if metodo == 'gmm':
        model = GMM(n_components=n_colors).fit(image_array_sample) #se crea un modelo GMM segun n_colors
        labels = model.predict(image_array)
        centers = model.means_

    return labels,centers #se retornan los labels y los centros

def calc_distance(): #funcion para calcular la suma de distancias intra-clusters
    partial_sum = 0 #se inicia suma parcial en 0
    for index in range(0, image_array.shape[0]):
        # se suman las normas de las diferencias entre un pixel y su respectivo centro
        partial_sum += np.linalg.norm((image_array[index] - centers[labels[index]]), axis=0)

    sums.append(partial_sum) #se guarda esta suma en sums
    return  sums

if __name__ == '__main__':

    #r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'
    path = input('ingrese path: ') #se pide al usuario que ingrese el path donde se encuentra la imagen
    image_name = input('image name: ') #r'bandera.png' #se pide al usuario que ingrese el nombre de la imagen
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file, 1) #se lee la imagen segun la ruta
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #se cambia el espacio de color (en realidad solo el orden)

    metodo = input('ingrese metodo: ') #se le pide al usuario que ingrese el metodo (k-means o GMM)

    image = np.array(image, dtype=np.float64) / 255 #se convierte a flotante y se normaliza la imagen
    rows, cols, ch = image.shape #de la imagen se obtienen las columnas, filas y canales
    assert ch == 3 #se asegura que sea una imagen a color
    image_array = np.reshape(image, (rows * cols, ch)) #se convierte la imagen a un arreglo unidimensional, mas cada componente tendra tres valores (RGB)
    print("Fitting model on a small sub-sample of the data..")
    image_array_sample = shuffle(image_array, random_state=0)[:10000] #se escogen 10000 muestras del arreglo

    #se muestra la imagen que se quiere segmentar
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)
    plt.show()

    sums = [] #variable que nos permitira guardar las sumas de distancias intra-clusters para cada valro de n_colors

    for n_colors in range(1,11): #ciclo para iterar el numero de clusters

        labels,centers = clustering(image, metodo) #se obtienen los labels y centros segun el metodo
        sums = calc_distance() #se calcula la suma de distancias intra-clusters
        recreate_image(centers, labels, rows, cols) #se muestra la imagen final segmentada

    #se grafica la suma de distancias intra-cluster vs numero de clusters/gaussianas
    plt.plot(list(range(1,11)),sums)
    plt.title('Sum of intra-cluster distances vs Number of colors')
    plt.xlabel('Number of colors')
    plt.ylabel('Sum of intra-cluster distances')
    plt.show()