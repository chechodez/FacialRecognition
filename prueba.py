import dlib
import cv2
import numpy as np
import os

predictor_path = os.path.abspath(os.getcwd()) + "/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
path = os.path.abspath(os.getcwd()) + "/images"
N = len(os.listdir(path))

lst_of_lst_distancia = []
etiquetas = []
puntos_interes = range(17, 66)
for i in range(100):
    if i < 50:
        etiquetas.append(0)  # Feli
    elif i >= 50 and i < 101:
        etiquetas.append(1)  # time

for contador in range(1, 100 + 1):
    lst_distancia = []
    frame = cv2.imread("./images2/image" + str(contador) + ".jpeg")
    frame = cv2.resize(frame, (320, 240))
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
    except IndexError:
        etiquetas.pop(contador - 1)

np.save("distancias", lst_of_lst_distancia)
np.save("etiquetas", etiquetas)
#########################################################################################
