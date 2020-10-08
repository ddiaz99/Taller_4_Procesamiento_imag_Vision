import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1

    # Display all results, alongside original image
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    plt.title('Quantized image ({} colors, method={})'.format(n_colors, metodo))
    plt.imshow(image_clusters)

    plt.show()

def clustering(image,metodo):
    if metodo == 'kmeans':
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = model.predict(image_array)
        centers = model.cluster_centers_

    if metodo == 'gmm':
        model = GMM(n_components=n_colors).fit(image_array_sample)
        labels = model.predict(image_array)
        centers = model.means_

    return labels,centers

def calc_distance():
    suma_parcial = 0
    for index in range(0, image_array.shape[0]):
        suma_parcial += np.linalg.norm((image_array[index] - centers[labels[index]]), axis=0)

    sumas.append(suma_parcial)
    return  sumas

if __name__ == '__main__':

    path = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'#input('ingrese path: ')
    image_name = r'bandera.png'#input('image name: ')
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    metodo = input('ingrese metodo: ')

    image = np.array(image, dtype=np.float64) / 255
    rows, cols, ch = image.shape
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch))
    print("Fitting model on a small sub-sample of the data..")
    image_array_sample = shuffle(image_array, random_state=0)[:10000]

    plt.figure(1)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)
    plt.show()

    sumas = []

    for n_colors in range(1,11):

        labels,centers = clustering(image, metodo)
        # Get labels for all points

        sumas = calc_distance()

        recreate_image(centers, labels, rows, cols)

    plt.plot(list(range(1,11)),sumas)
    plt.title('Sum of intra-cluster distances vs Number of colors')
    plt.xlabel('Number of colors')
    plt.ylabel('Sum of intra-cluster distances')
    plt.show()