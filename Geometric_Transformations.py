import numpy as np
import os
import cv2

def click_event(event, x, y, flags, params):
    global number_points
    if number_points < 3:
        if event == cv2.EVENT_RBUTTONDOWN:
            mouse_coordinates.append([x, y])
            number_points += 1
    else:
        number_points = 0
        cv2.destroyAllWindows()

def get_points():
    for i in range(0,len(images)):
        cv2.imshow(windowsNames[i],images[i])
        cv2.setMouseCallback(windowsNames[i], click_event)
        cv2.waitKey(0)

    pts1 = np.float32(mouse_coordinates[0:3]) #lena
    pts2 = np.float32(mouse_coordinates[3:6]) #lena_warped

    return pts1,pts2

def estimate_similarity(pts1,M_affine):

    point1_homogeneous = np.append(pts1[0], np.array([1]), axis=0)
    point2_homogeneous = np.append(pts1[1], np.array([1]), axis=0)
    point3_homogeneous = np.append(pts1[2], np.array([1]), axis=0)
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

    b = np.array([x1_prime,y1_prime,x2_prime,y2_prime,x3_prime,y3_prime])

    matrix1 = np.array([[x1,y1,1,0,0,0],[0,0,0,x1,y1,1],[x2,y2,1,0,0,0],[0,0,0,x2,y2,1],[x3,y3,1,0,0,0],[0,0,0,x3,y3,1]])
    matrix2 = matrix1.T

    A_1 = np.linalg.inv(matrix2.dot(matrix1))
    A_2 = A_1.dot(matrix2)
    A_3 = np.dot(A_2,b.T)

    a = A_3[0]
    b = A_3[1]
    d = A_3[3]
    e = A_3[4]

    theta = np.arctan(-b/e)
    s1 = a/np.cos(theta)
    s2 = e/np.cos(theta)
    tx = A_3[2]
    ty = A_3[5]

    print('aproximacion parametros de matriz de similitud')
    print(f'theta1 = {theta}')
    print(f's1 = {s1}')
    print(f's2 = {s2}')
    print(f'tx = {tx}')
    print(f'ty = {ty}')

    M_similarity = np.array([[a, b, tx],
                        [d, e, ty]])

    return M_similarity

def calc_error(pts1,M_similarity,pts2):
    pts = np.append(pts1.transpose(), np.array([[1, 1, 1]]), axis=0)
    pts_transform = M_similarity.dot(pts)
    pts_transform = pts_transform.transpose()

    error = np.linalg.norm(pts_transform - pts2, axis=1)
    print('Error calculated = ',error)

if __name__ == '__main__':
    mouse_coordinates = list()
    number_points = 0

    path1 = r'C:\Users\di-di\OneDrive\Escritorio\imagenes_vision'#input('ingrese path: ')
    image_name1 = r'lena.png'#input('image name: ')
    path_file1 = os.path.join(path1, image_name1)
    lena = cv2.imread(path_file1, 1)

    image_name2 = r'lena_warped.png'#input('image name: ')
    path_file2 = os.path.join(path1, image_name2)
    lena_warped = cv2.imread(path_file2, 1)

    windowsNames = ['lena','lena_warped']
    images = [lena,lena_warped]

    pts1,pts2 = get_points()

    M_affine = cv2.getAffineTransform(pts1, pts2)

    print('Affine transformation: ')
    print(M_affine)

    image_affine = cv2.warpAffine(lena, M_affine, lena.shape[:2])
    cv2.imshow("Affine", image_affine)


    M_similarity = estimate_similarity(pts1,M_affine)
    image_similarity = cv2.warpAffine(lena, M_similarity, lena.shape[:2])
    cv2.imshow('similarity', image_similarity)

    calc_error(pts1, M_similarity, pts2)

    cv2.waitKey(0)