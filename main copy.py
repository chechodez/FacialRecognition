import dlib
import cvzone
import cv2
import numpy as np
import os
import sys
from src.functions import deteccion_rostro, predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle


# import wiringpi
import time

# import RPi.GPIO as GPIO


# wiringpi.wiringPiSetup()
# wiringpi.pinMode(12, 1)

img_feliz = cv2.imread("Resources/img_feliz.png", cv2.IMREAD_UNCHANGED)
img_triste = cv2.imread("Resources/img_triste.png", cv2.IMREAD_UNCHANGED)

scale_percent = 22
width = int(img_feliz.shape[1] * scale_percent / 100)
height = int(img_feliz.shape[0] * scale_percent / 100)

img_feliz = cv2.resize(img_feliz, (width, height))

width = int(img_triste.shape[1] * scale_percent / 100)
height = int(img_triste.shape[0] * scale_percent / 100)

img_triste = cv2.resize(img_triste, (width, height))

predictor_path = os.path.abspath(os.getcwd()) + "/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
path = os.path.abspath(os.getcwd()) + "/images"
N = len(os.listdir(path))


#########################################################################################
lst_of_lst_distancia = np.load("distancias.npy")
etiquetas = np.load("etiquetas.npy")

# ML
X_train, X_test, y_train, y_test = train_test_split(
    lst_of_lst_distancia, etiquetas, test_size=0.2, random_state=0
)
# Se escalan las distancias
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
training_acc = []
lst_knn = []
k_range = range(1, int(np.sqrt(len(y_train))))
distance = "manhattan"
for k in k_range:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric=distance,
        metric_params=None,
        algorithm="brute",
    )
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    # training_acc.append(knn.score(X_train, y_train))
    lst_knn.append(knn)
    print(y_test, y_predicted)

# knn = lst_knn[training_acc.index(max(training_acc))]
# Se realiza la comparacion por genero
lst_of_lst_distancia_men = np.load("distancias_men.npy")
etiquetas_men = np.load("etiquetas_men.npy")
X_men = knn.predict(lst_of_lst_distancia_men)
f1_acurracy_men = f1_score(X_men, etiquetas_men)

contador_men = 0
for index, item in enumerate(X_men):
    if item == etiquetas_men[index]:
        contador_men += 1
porcentaje_accuracy_men = contador_men * 100 / len(lst_of_lst_distancia_men)
print(
    "El porcentaje de accuracy para del sistema para los hombres es de: {}% y tiene F1 score de: {}".format(
        porcentaje_accuracy_men, f1_acurracy_men
    )
)

lst_of_lst_distancia_women = np.load("distancias_women.npy")
etiquetas_women = np.load("etiquetas_women.npy")
X_women = knn.predict(lst_of_lst_distancia_women)

f1_acurracy_women = f1_score(X_women, etiquetas_women)
contador_women = 0
for index, item in enumerate(X_women):
    if item == etiquetas_women[index]:
        contador_women += 1

porcentaje_accuracy_women = contador_women * 100 / len(lst_of_lst_distancia_women)
print(
    "El porcentaje de accuracy para del sistema para las mujeres es de: {}% y tiene F1 score de: {}".format(
        porcentaje_accuracy_women, f1_acurracy_women
    )
)
a = 1

cap = cv2.VideoCapture(0)
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(24, GPIO.OUT)
# rojo = GPIO.PWM(24, 100)
# rojo.start(100)
contador = 70
timer_1 = time.perf_counter()
timer_2 = timer_1

# Ciclo
while True:
    _, frame = cap.read()
    # frame = cv2.imread("img_pruebas/image2.jpeg")
    frame = cv2.resize(frame, (320, 240))
    [frame_copy, frame_copy_draw] = [frame.copy(), frame.copy()]
    dist = []
    dist2 = []
    try:
        [points, shape, flag_face] = deteccion_rostro(
            frame=frame, detector=detector, predictor=predictor
        )
        if flag_face:
            for (x, y) in points:
                dist.append(
                    ((x - points[16][0]) * 2 + (y - points[16][1]) * 2) ** 1 / 2
                )
                cv2.circle(frame_copy_draw, (x, y), 2, (0, 0, 255), -1)

            tolerancia = 25
            frame[
                shape.part(19).y - tolerancia : shape.part(8).y,
                shape.part(0).x : shape.part(16).x,
            ] = 0
            frame_copy_draw = predict(
                knn=knn,
                dist=dist,
                scaler=scaler,
                points=points,
                frame_copy_draw=frame_copy_draw,
                img_feliz=img_feliz,
                img_triste=img_triste,
            )
        # Segundo rostro
        [points, shape, flag_face] = deteccion_rostro(frame, detector, predictor)
        if flag_face:
            for (x, y) in points:
                dist2.append(
                    ((x - points[16][0]) * 2 + (y - points[16][1]) * 2) ** 1 / 2
                )
                cv2.circle(frame_copy_draw, (x, y), 2, (0, 0, 255), -1)
            frame_copy_draw = predict(
                knn=knn,
                dist=dist2,
                scaler=scaler,
                points=points,
                frame_copy_draw=frame_copy_draw,
                img_feliz=img_feliz,
                img_triste=img_triste,
            )
        # Se pone la imagen original debajo
        scale_percent = 20
        width = int(frame_copy.shape[1] * scale_percent / 100)
        height = int(frame_copy.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame_copy = cv2.resize(frame_copy, dim, interpolation=cv2.INTER_AREA)
        frame_copy_draw[
            frame.shape[0] - frame_copy.shape[0] - 1 : frame.shape[0] - 1,
            frame.shape[1] - frame_copy.shape[1] - 1 : frame.shape[1] - 1,
            :,
        ] = frame_copy

    except:
        pass
    cv2.imshow("Video", cv2.resize(frame_copy_draw, (1280, 720)))
    cv2.waitKey(5)
