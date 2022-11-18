import dlib
import cv2
import numpy as np
import os
import sys
import pickle


def lectura(path):
    # data_dir_list = os.listdir(path)
    data_dir_list = ["happy", "sad", "surprise"]
    img_data_list = []
    lst_count = []
    counter = 0
    for dataset in data_dir_list:
        img_list = os.listdir(path + "/" + dataset)
        for img in img_list:
            input_img = cv2.imread(path + "/" + dataset + "/" + img)
            # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # input_img_resize = cv2.resize(input_img, (48, 48))
            img_data_list.append(input_img)
            counter += 1
        lst_count.append(counter)
    return img_data_list, lst_count


predictor_path = os.path.abspath(os.getcwd()) + "\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
lst_of_lst_distancia = []
etiquetas = []
path = os.path.abspath(os.getcwd()) + "\\database\\fer2013\\train"
[images, lst_count] = lectura(path)
N = len(images)
# cv2.imshow('a',images[731])
# cv2.waitKey(0)

for i in range(N):
    if i < lst_count[0]:
        etiquetas.append(0)  # Feli
    elif i >= lst_count[0] and i < lst_count[1]:
        etiquetas.append(1)  # tite
    elif i >= lst_count[1] and i < lst_count[2]:
        etiquetas.append(2)  # solplesa

contador_etiquetas = 0

for contador in range(N):
    lst_distancia = []
    frame = images[contador]
    # frame = cv2.resize(frame, (320, 240))
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    try:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            old_points = np.zeros((49, 2), dtype=np.int32)
            for i in range(0, 49):
                old_points[i] = (shape.part(i + 17).x, shape.part(i + 17).y)
            for (x, y) in old_points:
                lst_distancia.append(
                    ((x - old_points[16][0]) * 2 + (y - old_points[16][1]) * 2) ** 1 / 2
                )
            lst_of_lst_distancia.append(lst_distancia)
        contador_etiquetas += 1
    except IndexError:
        etiquetas.pop(contador_etiquetas - 1)

np.save("distancias", lst_of_lst_distancia)
np.save("etiquetas", etiquetas)
