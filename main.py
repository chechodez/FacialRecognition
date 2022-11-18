import dlib
import cvzone
import cv2
import numpy as np
import os
import sys
from src.functions import deteccion_rostro, predict, get_model_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
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
img_sorpresa = cv2.imread("Resources/img_sorpresa.png", cv2.IMREAD_UNCHANGED)

scale_percent = 8
width = int(img_feliz.shape[1] * scale_percent / 100)
height = int(img_feliz.shape[0] * scale_percent / 100)

img_feliz = cv2.resize(img_feliz, (width, height))

width = int(img_triste.shape[1] * scale_percent / 100)
height = int(img_triste.shape[0] * scale_percent / 100)

img_triste = cv2.resize(img_triste, (width, height))

width = int(img_sorpresa.shape[1] * scale_percent / 100)
height = int(img_sorpresa.shape[0] * scale_percent / 100)

img_sorpresa = cv2.resize(img_sorpresa, (width, height))

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
dict_knn = {}
k_range = range(1, int(np.sqrt(len(y_train))))
distance = "manhattan"
# for k in k_range:
#     knn = KNeighborsClassifier(
#         n_neighbors=k,
#         weights="distance",
#         metric=distance,
#         metric_params=None,
#         algorithm="brute",
#     )
#     knn.fit(X_train, y_train)
#     y_predicted = knn.predict(X_test)
#     # training_acc.append(knn.score(X_train, y_train))

#     dict_knn[k] = matthews_corrcoef(y_test, y_predicted)
#     print(y_test, y_predicted)
# max(dict_knn, key=dict_knn.get)  # se toma el valor maximo
# knn = knn = KNeighborsClassifier(
#     n_neighbors=13,
#     weights="distance",
#     metric=distance,
#     metric_params=None,
#     algorithm="brute",
# ) # dos emociones
knn = knn = KNeighborsClassifier(
    n_neighbors=35,
    weights="distance",
    metric=distance,
    metric_params=None,
    algorithm="brute",
)  # tres emociones
knn.fit(X_train, y_train)

# Se realiza la comparacion por genero
lst_of_lst_distancia_men = np.load("distancias_men.npy")
etiquetas_men = np.load("etiquetas_men.npy")
confusion_matrix_men = get_model_metrics(
    knn, lst_of_lst_distancia_men, etiquetas_men, "hombres"
)


lst_of_lst_distancia_women = np.load("distancias_women.npy")
etiquetas_women = np.load("etiquetas_women.npy")
confusion_matrix_women = get_model_metrics(
    knn, lst_of_lst_distancia_women, etiquetas_women, "mujeres"
)

confusion_matrix = get_model_metrics(knn, X_test, y_test, "general")

a = 1

cap = cv2.VideoCapture(0)
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(24, GPIO.OUT)
# rojo = GPIO.PWM(24, 100)
# rojo.start(100)
contador = 70
timer_1 = time.perf_counter()
timer_2 = timer_1
pTime = 0
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
            # cv2.circle(frame_copy_draw, (x, y), 2, (0, 0, 255), -1)

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
                img_sorpresa=img_sorpresa,
            )
        # Segundo rostro
        [points, shape, flag_face] = deteccion_rostro(frame, detector, predictor)
        if flag_face:
            for (x, y) in points:
                dist2.append(
                    ((x - points[16][0]) * 2 + (y - points[16][1]) * 2) ** 1 / 2
                )
                # cv2.circle(frame_copy_draw, (x, y), 2, (0, 0, 255), -1)
            frame_copy_draw = predict(
                knn=knn,
                dist=dist2,
                scaler=scaler,
                points=points,
                frame_copy_draw=frame_copy_draw,
                img_feliz=img_feliz,
                img_triste=img_triste,
                img_sorpresa=img_sorpresa,
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
    cTime = time.time()

    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        frame_copy_draw,
        f"FPS: {int(fps)}",
        (150, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 0),
        1,
    )
    cv2.imshow("Video", cv2.resize(frame_copy_draw, (1280, 720)))
    cv2.waitKey(5)
