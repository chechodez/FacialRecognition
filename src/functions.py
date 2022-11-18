import dlib
import cvzone
import cv2
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def get_model_metrics(model, lst_of_lst, etiquetas, gender=None):
    X_men = model.predict(lst_of_lst)
    matheus_acurracy_men = matthews_corrcoef(etiquetas, X_men)
    f1_accuracy_men = f1_score(
        etiquetas, X_men, average="weighted"
    )  # para dos emociones
    # f1_accuracy_men = 0
    confusion_matrix_men = confusion_matrix(etiquetas, X_men)

    contador_men = 0
    # for index, item in enumerate(X_men):
    #     if item == etiquetas[index]:
    #         contador_men += 1
    # porcentaje_accuracy_men = contador_men * 100 / len(lst_of_lst)
    porcentaje_accuracy_men = accuracy_score(etiquetas, X_men)
    print(
        "El porcentaje de accuracy para del sistema para {}  es de: {}% y coeficiente de matthews score de: {}, F1 score de {}".format(
            gender, porcentaje_accuracy_men, matheus_acurracy_men, f1_accuracy_men
        )
    )
    return confusion_matrix_men


def deteccion_rostro(frame, detector, predictor):
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    points = []
    if len(dets) != 0:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            points = np.zeros((49, 2), dtype=np.int32)
            for i in range(0, 49):
                points[i] = (shape.part(i + 17).x, shape.part(i + 17).y)
            flag_face = True
    else:
        shape = []
        flag_face = False
    return [points, shape, flag_face]


def predict(
    knn, dist, scaler, points, frame_copy_draw, img_feliz, img_triste, img_sorpresa
):
    arreglo = np.array(dist)
    predict_item = scaler.transform(arreglo.reshape(1, arreglo.shape[0]))
    y_predict = knn.predict(predict_item)
    if y_predict == 0:
        # print(contador)
        # if (timer_1 - timer_2) > 0.001:

        #     contador = contador+5
        #     if contador == 100:
        #         contador = 70
        #     rojo.ChangeDutyCycle(100 - contador)
        #     timer_2 = timer_1
        # else:
        #     timer_1 = time.perf_counter()
        frame_copy_draw = cvzone.overlayPNG(
            frame_copy_draw, img_feliz, [points[1][0] + 10, points[1][1] - 40]
        )
    elif y_predict == 1:
        frame_copy_draw = cvzone.overlayPNG(
            frame_copy_draw, img_triste, [points[1][0] + 10, points[1][1] - 40]
        )
    elif y_predict == 2:
        frame_copy_draw = cvzone.overlayPNG(
            frame_copy_draw, img_sorpresa, [points[1][0] + 10, points[1][1] - 40]
        )
    return frame_copy_draw, y_predict
    # rojo.ChangeDutyCycle(0)
