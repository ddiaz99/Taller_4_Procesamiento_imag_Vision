import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

def clustering(image,metodo):
    if metodo == 'kmeans':
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    if metodo == 'gmm':
        model = GMM(n_components=n_colors).fit(image_array_sample)

    return model
    print("done in %0.3fs." % (time() - t0))

#path_file = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision\lena.jpg'
path = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'#input('ingrese path: ')
image_name = r'bandera.png'#input('image name: ')
path_file = os.path.join(path, image_name)
image = cv2.imread(path_file, 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
metodo = input('ingrese metodo: ')

# cv2.imshow('lena',image)
# cv2.waitKey(0)

image = np.array(image, dtype=np.float64) / 255
rows, cols, ch = image.shape
assert ch == 3
image_array = np.reshape(image, (rows * cols, ch))
print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:10000]

n_colors = 6

plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(image)

for n_colors in range(1,11):

    model = clustering(image, metodo)
    # Get labels for all points
    print("Predicting color indices on the full image (GMM)")
    t0 = time()
    if metodo == 'gmm':
        labels = model.predict(image_array)
        centers = model.means_
    else:
        labels = model.predict(image_array)
        centers = model.cluster_centers_
    print("done in %0.3fs." % (time() - t0))

    # Display all results, alongside original image

    plt.figure(2)
    plt.clf()
    plt.axis('off')
    plt.title('Quantized image ({} colors, method={})'.format(n_colors, metodo))
    plt.imshow(recreate_image(centers, labels, rows, cols))

    plt.show()