import numpy as np
import os
import cv2

mouse_coordinates = list()
number_points = 0

def click_event(event, x, y, flags, params):
    global number_points
    if number_points < 3:
        if event == cv2.EVENT_RBUTTONDOWN:
            mouse_coordinates.append([x, y])
            print(mouse_coordinates)
            number_points += 1
    else:
        number_points = 0
        cv2.destroyAllWindows()

path1 = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'#input('ingrese path: ')
image_name1 = r'lena.jpg'#input('image name: ')
path_file1 = os.path.join(path1, image_name1)
lena = cv2.imread(path_file1, 1)

image_name2 = r'lena_warped.png'#input('image name: ')
path_file2 = os.path.join(path1, image_name2)
lena_warped = cv2.imread(path_file2, 1)

windowsNames = ['lena','lena_warped']
images = [lena,lena_warped]


for i in range(0,len(images)):
    cv2.imshow(windowsNames[i],images[i])
    cv2.setMouseCallback(windowsNames[i], click_event)
    cv2.waitKey(0)

pts1 = np.float32(mouse_coordinates[0:3])
pts2 = np.float32(mouse_coordinates[3:6])

M_affine = cv2.getAffineTransform(pts1, pts2)
print(M_affine)
image_affine = cv2.warpAffine(lena, M_affine, lena.shape[:2])

cv2.imshow("Image", image_affine)
cv2.waitKey(0)










