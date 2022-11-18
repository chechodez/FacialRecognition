import dlib
import cvzone
import cv2
import numpy as np
import os
import sys
from src.functions import deteccion_rostro, predict, get_model_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pickle
import pandas as pd
from datetime import datetime


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

kernel = ["poly", "rbf", "sigmoid"]
regularizacion = [0.01, 0.1, 1, 10, 100]
grado_poly = range(1, 20)
my_dict = {}
# for i in kernel:
#     for j in regularizacion:
#         for k in grado_poly:
#             svm_model = SVC(C=j, kernel=i, degree=k)
#             svm_model.fit(X_train, y_train)
#             my_dict[i, j, k] = matthews_corrcoef(y_test, svm_model.predict(X_test))
#             print("voy en la iteracion {} {} {}".format(i, j, k))

# svm_model = SVC(C=10, kernel=kernel[0], degree=1)  # Es para 2 emociones
svm_model = SVC(C=10, kernel=kernel[0], degree=1)  # Es para 3 emociones
svm_model.fit(X_train, y_train)
# Se realiza la comparacion por genero
lst_of_lst_distancia_men = np.load("distancias_men.npy")
etiquetas_men = np.load("etiquetas_men.npy")
confusion_matrix_men = get_model_metrics(
    svm_model, lst_of_lst_distancia_men, etiquetas_men, "hombres"
)


lst_of_lst_distancia_women = np.load("distancias_women.npy")
etiquetas_women = np.load("etiquetas_women.npy")
confusion_matrix_women = get_model_metrics(
    svm_model, lst_of_lst_distancia_women, etiquetas_women, "mujeres"
)
a = 1
confusion_matrix = get_model_metrics(svm_model, X_test, y_test, "general")


cap = cv2.VideoCapture(0)
# Ciclo
df_info = pd.DataFrame(columns=["TimeStamp", "Persona 1", "Persona 2"])
previous_prediction1 = None
previous_prediction2 = None
while True:
    _, frame = cap.read()
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord("q"):
        df_info.to_excel("historicos/df_info.xlsx")
        break
    # frame = cv2.imread("img_pruebas/tres_felices.jpg")q
    # frame = cv2.imread("img_pruebas/image2.jpeg")
    frame = cv2.resize(frame, (320, 240))
    [frame_copy, frame_copy_draw] = [frame.copy(), frame.copy()]
    dist = []
    dist2 = []
    dist3 = []
    df_aux = pd.DataFrame(columns=["TimeStamp", "Persona 1", "Persona 2"])
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
            [frame_copy_draw, prediction_1] = predict(
                knn=svm_model,
                dist=dist,
                scaler=scaler,
                points=points,
                frame_copy_draw=frame_copy_draw,
                img_feliz=img_feliz,
                img_triste=img_triste,
                img_sorpresa=img_sorpresa,
            )
            if prediction_1 != previous_prediction1:
                previous_prediction1 = prediction_1
                df_aux = df_aux.append(
                    pd.DataFrame(
                        data={
                            "TimeStamp": datetime.now(),
                            "Persona 1": prediction_1,
                            "Persona 2": None,
                        }
                    ),
                    ignore_index=True,
                )

        # Segundo rostro
        [points, shape, flag_face] = deteccion_rostro(frame, detector, predictor)
        if flag_face:
            for (x, y) in points:
                dist2.append(
                    ((x - points[16][0]) * 2 + (y - points[16][1]) * 2) ** 1 / 2
                )
                # cv2.circle(frame_copy_draw, (x, y), 2, (0, 0, 255), -1)
            frame[
                shape.part(19).y - tolerancia : shape.part(8).y,
                shape.part(0).x : shape.part(16).x,
            ] = 0
            [frame_copy_draw, prediction_2] = predict(
                knn=svm_model,
                dist=dist2,
                scaler=scaler,
                points=points,
                frame_copy_draw=frame_copy_draw,
                img_feliz=img_feliz,
                img_triste=img_triste,
                img_sorpresa=img_sorpresa,
            )
            if prediction_2 != previous_prediction2:
                previous_prediction2 = prediction_2
                df_aux["Persona 2"] = prediction_2

        if not (df_aux.empty):
            df_info = df_info.append(df_aux, ignore_index=True)

        # Tercer rostro
        # [points, shape, flag_face] = deteccion_rostro(frame, detector, predictor)
        # if flag_face:
        #     for (x, y) in points:
        #         dist3.append(
        #             ((x - points[16][0]) * 2 + (y - points[16][1]) * 2) ** 1 / 2
        #         )
        #         # cv2.circle(frame_copy_draw, (x, y), 2, (0, 0, 255), -1)
        #     frame[
        #         shape.part(19).y - tolerancia : shape.part(8).y,
        #         shape.part(0).x : shape.part(16).x,
        #     ] = 0
        #     frame_copy_draw = predict(
        #         knn=svm_model,
        #         dist=dist3,
        #         scaler=scaler,
        #         points=points,
        #         frame_copy_draw=frame_copy_draw,
        #         img_feliz=img_feliz,
        #         img_triste=img_triste,
        #         img_sorpresa=img_sorpresa,
        #     )
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
    # cTime = time.time()

    # fps = 1 / (cTime - pTime)
    # pTime = cTime

    # cv2.putText(
    #     frame_copy_draw,
    #     f"FPS: {int(fps)}",
    #     (150, 50),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     1,
    #     (255, 255, 0),
    #     1,
    # )
    cv2.imshow("Video", cv2.resize(frame_copy_draw, (1280, 720)))
    cv2.waitKey(5)
