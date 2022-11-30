import math
import os
import shutil
import sys
import traceback
# import traceback
from datetime import datetime

# from copy import deepcopy

import cv2
import geopy.distance
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt

# +info
# https://www.gpsworld.com/what-exactly-is-gps-nmea-data/
# https://www.earthpoint.us/convert.aspx


# Distancia máxima en metros: diagonal de la imagen
# 0.2098: relación cm/px calculada para imágenes corregidas de cámara 0
max_distance = 0.2098 * math.sqrt(2448 ** 2 + 968 ** 2) / 100  # en metros

show_img = True
img_size = [0] * 2
ratio = 1
distance_type = "leg"
distance_operation = "avg"
scale = 1
max_dist = 0

min_keypoints = 5
image_group = 10


def fix_lane_coord(p_lane: list) -> list:
    """
    Corrige coordenadas repetidas consecutivas, calculando el valor medio entre coordenadas

    :param p_lane: lista con la información del carril
    :return: lista corregida
    """
    # Si las coordenadas están repetidas, las corregimos
    for k in range(1, len(p_lane) - 2):
        lat_0, lat_1, lat_2 = p_lane[k - 1][-2], p_lane[k][-2], p_lane[k + 1][-2]  # latitud
        lon_0, lon_1, lon_2 = p_lane[k - 1][-1], p_lane[k][-1], p_lane[k + 1][-1]  # longitud

        if lat_1 == lat_0 and lon_1 == lon_0:
            p_lane[k][-2] = abs(lat_0 - lat_2) / 2 + min([lat_0, lat_2])
            p_lane[k][-1] = abs(lon_0 - lon_2) / 2 + min([lon_0, lon_2])

    return p_lane


def prepare_data(p_folder_l1, p_folder_l2, p_reverse=0):
    """
    Prepara la información a partir de dos carpetas de entrada (L1 y L2)

    :param p_folder_l2:
    :param p_folder_l1:
    :param p_reverse: 0 (no lane reversed), 1 (lane 1), 2 (lane 2), 3 (both lanes)
    :return:
    """
    path_prepared_1 = "./res/input/prepared_lane_1"
    path_prepared_2 = "./res/input/prepared_lane_2"

    list_lane_1 = []
    list_lane_2 = []

    # Vacía las carpetas de salida
    for file in os.listdir(path_prepared_1):
        os.remove(path_prepared_1 + "/" + file) if file.endswith(".jpg") else 0

    for file in os.listdir(path_prepared_2):
        os.remove(path_prepared_2 + "/" + file) if file.endswith(".jpg") else 0

    # Agrupa los ficheros de imagen de cada carpeta
    for file in os.listdir(p_folder_l1):
        if file.endswith(".jpg"):
            # file = file.split("_")[4]
            list_lane_1.append(p_folder_l1 + "/" + file)

    for file in os.listdir(p_folder_l2):
        if file.endswith(".jpg"):
            # file = file.split("_")[4]
            list_lane_2.append(p_folder_l2 + "/" + file)

    # Número máximo de caracteres en texto
    chr_max = len(str(max([len(list_lane_1), len(list_lane_2)])))

    # Carril 1
    for p_idx, element in enumerate(list_lane_1):
        # Asigna un nombre numérico a la imagen, e invierte el orden si el carril está invertido
        temp_name = str(len(list_lane_1) - p_idx - 1).zfill(chr_max) if p_reverse in [1, 3] else str(p_idx).zfill(
            chr_max)
        list_lane_1[p_idx] = [element, path_prepared_1 + "/" + temp_name + ".jpg"]

        # Copia la imagen y la gira si está invertida
        if p_reverse in [1, 3]:
            cv2.imwrite(list_lane_1[p_idx][1], cv2.rotate(cv2.imread(list_lane_1[p_idx][0]), cv2.ROTATE_180))
        else:
            shutil.copy(list_lane_1[p_idx][0], list_lane_1[p_idx][1])

        sys.stdout.write("\r- Preparing lane 1 (%.2f %%)" % (100 * (p_idx + 1) / len(list_lane_1)))

    print("")

    # Carril 2
    for p_idx, element in enumerate(list_lane_2):
        # Asigna un nombre numérico a la imagen, e invierte el orden si el carril está invertido
        temp_name = str(len(list_lane_2) - p_idx - 1).zfill(chr_max) if p_reverse in [2, 3] else str(p_idx).zfill(
            chr_max)
        list_lane_2[p_idx] = [element, path_prepared_2 + "/" + temp_name + ".jpg"]

        # Copia la imagen y la gira si está invertida
        if p_reverse in [2, 3]:
            cv2.imwrite(list_lane_2[p_idx][1], cv2.rotate(cv2.imread(list_lane_2[p_idx][0]), cv2.ROTATE_180))
        else:
            shutil.copy(list_lane_2[p_idx][0], list_lane_2[p_idx][1])

        sys.stdout.write("\r- Preparing lane 2 (%.2f %%)" % (100 * (p_idx + 1) / len(list_lane_2)))

    print("")

    # Aplica reversión a las listas si las imágenes estaban ordenadas de forma inversa
    if p_reverse in [1, 3]:
        list_lane_1.reverse()

    if p_reverse in [2, 3]:
        list_lane_2.reverse()

    list_lane_1 = append_coordinates(list_lane_1)
    list_lane_2 = append_coordinates(list_lane_2)

    return list_lane_1, list_lane_2


def transform_coordinates(p_coordinate: str) -> list:
    """
    Transforma el formato de coordenadas de GPS NMEA a LAT-LON en grados
    NMEA -> Dec Mins: ddmm.mmmmm(+N/-S) dddmm.mmmmm(+E/-W)

    :param p_coordinate: coordenadas NMEA
    :return: latitud y longitud en grados
    """

    # Discrimina los datos separados por un espacio
    lat, lon = p_coordinate.split(" ")

    # Separa la información (grados, minutos, orientación)
    lat = [lat[0:2], lat[2:-1], lat[-1]]
    lon = [lon[0:3], lon[3:-1], lon[-1]]

    # Aplica la transformación a grados (el signo depende de la orientación)
    lat = (float(lat[0]) + float(lat[1]) / 60) * (-1 if lat[2] in ["S"] else 1)
    lon = (float(lon[0]) + float(lon[1]) / 60) * (-1 if lon[2] in ["W"] else 1)

    return [lat, lon]


def append_coordinates(p_lane) -> list:
    """
    Obtiene las coordenadas de un tramo de carril por su nombre, las transforma en LAT-LON y las asocia a este

    :param p_lane: tramo de carril
    :return: tramo de carril con coordenadas asociadas
    """
    for p_idx, element in enumerate(p_lane):
        name = element[0].split("/")[-1]
        lat, lon = transform_coordinates(" ".join([name.split("_")[5], name.split("_")[6]]))

        p_lane[p_idx].extend([lat, lon])

    p_lane = fix_lane_coord(p_lane)

    return p_lane


def match_coordinates(p_lane_1: list, p_lane_2: list) -> list:
    """
    Por cada imagen de L1, identifica una coincidencia en L2 y la asocia, así como su distancia (en metros)

    :param p_lane_1: lista con datos de L1
    :param p_lane_2: lista con datos de L2
    :return: lista unificada de L1 y coincidencias de L2
    """
    p_array = []
    bad_match = 0

    # Por cada elemento de L1
    for idx_i, lane_i in enumerate(p_lane_1):

        # Reiniciamos la distancia y asociación más cercanas
        nearest_dst, nearest_img = None, None

        # Por cada elemento de L2
        for idx_j, lane_j in enumerate(p_lane_2):

            # Calculamos la distancia entre elementos (en metros)
            j_distance = geopy.distance.geodesic(lane_i[2:4], lane_j[2:4]).m

            # Si es más cercana que la almacenada previamente, la asocia
            if nearest_dst is None or nearest_dst > j_distance:
                nearest_dst = j_distance
                nearest_img = [idx_j, lane_j]  # almacena el índice en lista y el elemento

        # Incorporamos los datos del elemento L1
        p_array.append(lane_i)

        # Si se encontró un valor de L2 relacionado, se asocia. En caso contrario, se deja en nulo
        if nearest_dst is not None and nearest_dst < max_distance:
            p_array[-1].extend(nearest_img[1])
            p_array[-1].append(nearest_dst)

            # Elimina el elemento de L2 (por índice) para evitar duplicidades
            p_lane_2.pop(nearest_img[0])
        else:
            p_array[-1].extend([None] * (len(lane_i) + 1))
            bad_match += 1

        # Salida por pantalla del estado del proceso
        sys.stdout.write("\r- Matching coordinates (%.2f %%): %d/%d (ok/ko)" % (
            100 * (idx_i + 1) / len(p_lane_1), idx_i + 1 - bad_match, bad_match))

    print("")

    return p_array  # pd.DataFrame(p_array, columns=["lane_1_src", "lane_1_prep", "lane_1_lat",
    # "lane_1_lon", "lane_2_src", "lane_2_prep", "lane_2_lat", "lane_2_lon", "dist"])


def get_distance(p_good, p_kp1, p_kp2, p_max, p_type):
    """

    :param p_good:
    :param p_kp1:
    :param p_kp2:
    :param p_max:
    :param p_type:
    :return:
    """
    for k, element in enumerate(p_good):
        p_good[k] = element[0]

    # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python

    list_kp1 = []
    list_kp2 = []

    for x in p_good:
        img1_idx = x.queryIdx
        img2_idx = x.trainIdx

        (x_1, y_1) = p_kp1[img1_idx].pt
        (x_2, y_2) = p_kp2[img2_idx].pt

        list_kp1.append((x_1, y_1))
        list_kp2.append((x_2 + img_size[1], y_2))

    dist = []

    for k in range(0, len(list_kp1)):
        x = np.abs(list_kp2[k][0] - list_kp1[k][0])
        y = np.abs(list_kp2[k][1] - list_kp1[k][1])

        # Opciones para el cálculo de la distancia

        if p_type == "hypotenuse":  # Hipotenusa (Teorema de Pitágoras)
            distance = np.sqrt(x ** 2 + y ** 2)

        elif p_type == "leg_x":  # Cateto x
            distance = x

        elif p_type == "leg_y":  # Cateto y
            distance = x

        else:  # Promedio entre hipotenusa y cateto x
            distance = np.average([np.sqrt(x ** 2 + y ** 2), x])

        if distance <= p_max:
            dist.append(distance)
        # print(list_kp1[i], " -> ", list_kp2[i], " -> ", np.sqrt(x ** 2 + y ** 2))

    if len(dist) > 0:
        if distance_operation == "avg":
            return np.average(dist)
        elif distance_operation == "max":
            return np.max(dist)
        elif distance_operation == "min":
            return np.min(dist)

    return None


def img_filter(p_img, p_op=None):
    gamma = math.log(0.5 * 255) / math.log(np.mean(p_img))
    if p_op == "gamma":
        p_img = np.power(p_img, gamma).clip(0, 255).astype(np.uint8)

    elif p_op == "hist":
        p_img = cv2.equalizeHist(p_img)

    elif p_op == "hg":
        p_img = np.power(cv2.equalizeHist(p_img), gamma).clip(0, 255).astype(np.uint8)

    elif p_op == "gh":
        p_img = cv2.equalizeHist(np.power(p_img, gamma).clip(0, 255).astype(np.uint8))

    return p_img


def draw_overlap_transversal(p_row, p_idx="0"):
    """
    Dibuja el solape entre carriles
    :param p_row:
    :param p_idx:
    :return:
    """

    p_match = p_row["MATCH"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    p_i1 = cv2.imread(p_row["1_P"])

    if p_row["2_P"] == "":
        p_i2 = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    else:
        p_i2 = cv2.imread(p_row["2_P"])

    p_i1 = p_i1[0:p_i1.shape[0], 0:p_i1.shape[1] - int(p_row["DIST"])]  # height, width

    p_i4 = np.concatenate((p_i1, p_i2), axis=1)

    r_i4 = np.shape(p_i4)[1] / np.shape(p_i4)[0]
    p_i4 = cv2.resize(p_i4, (int(r_i4 * 640), 640), interpolation=cv2.INTER_AREA)

    text_1 = "Imagen carril 1: %s" % p_row["1_S"].split("/")[-1]
    text_2 = "Imagen carril 2: %s" % p_row["2_S"].split("/")[-1] if p_row["2_S"] != "" else "Imagen carril 2: -"
    text_3 = "Distancia media: %d px" % p_row["DIST"]

    cv2.putText(p_i4, text_1, (50, 150), font, font_scale, font_color, thickness, line_type)
    cv2.putText(p_i4, text_2, (50, 200), font, font_scale, font_color, thickness, line_type)
    cv2.putText(p_i4, text_3, (50, 250), font, font_scale, font_color, thickness, line_type)

    if p_match == "MATCHED":
        cv2.circle(p_i4, (50, 50), 25, (0, 255, 0), -1)
    else:
        cv2.circle(p_i4, (50, 50), 25, (0, 0, 255), -1)

    cv2.imwrite("./res/output/video/img" + p_idx + ".jpg", p_i4)


def draw_knn(p_f1, p_f2, p_i3, p_dist, p_match):
    """
    Dibuja el resultado del método SIFT (Knn)
    :param p_i3: imágenes adyacentes con líneas de coincidencia Knn
    :param p_f1: nombre de la imagen de L1
    :param p_f2: nombre de la imagen de L2
    :param p_dist: distancia calculada de solape
    :param p_match:
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    p_i3 = cv2.resize(p_i3, (int(ratio * 2 * 320), 320), interpolation=cv2.INTER_AREA)

    cv2.putText(p_i3, "L1-%s | L2-%s" % (p_f1, p_f2), (30, 60), font, font_scale, font_color, thickness,
                line_type)

    text = "%.0f px LIM | %.0f px AVG" % (
        (max_dist / scale), (p_dist / scale) if p_dist is not None else -1)

    cv2.putText(p_i3, text, (int(150 * ratio) + 30, 30), font, font_scale, font_color, thickness,
                line_type)

    cv2.putText(p_i3, "Min. kp: %d | Keypoints: %d" % (min_keypoints, p_match),
                (int(150 * ratio) + 30, 60), font, font_scale, font_color, thickness, line_type)

    cv2.imwrite("./res/output/match/" + p_f1 + "-" + p_f2 + "_" + str(int(p_dist / scale)) + ".jpg", p_i3)


def sift_function(p_lane_1, p_lane_2):
    """
    
    :param p_lane_1: 
    :param p_lane_2:
    :return: 
    """

    global max_dist
    avg_dist = None

    try:
        img_1 = cv2.imread(p_lane_1)
        img_2 = cv2.imread(p_lane_2)

        img_1 = cv2.resize(img_1, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        img_2 = cv2.resize(img_2, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)

        img1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        f1 = p_lane_1[p_lane_1.rfind("/") + 1:-4]
        f2 = p_lane_2[p_lane_2.rfind("/") + 1:-4]

        # Initiate SIFT creator
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default parameters
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply radio test
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])

        # Graph base
        img3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow("", cv2.resize(img3, (int(ratio * 2 * 160), 160), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)

        # Distances
        matches = np.asarray(good)
        if len(matches[:]) >= min_keypoints:

            avg_distance = get_distance(good, kp1, kp2, p_max=max_dist, p_type="leg_x")

            if avg_distance is not None:
                avg_dist = int(np.average(avg_distance))

                draw_knn(f1, f2, img3, avg_distance, len(matches[:]))

        return avg_dist

    except Exception:
        print(traceback.format_exc())
        return avg_dist


def match_images(p_lane_1, p_lane_2):
    # ----------------------------------------------------------------------

    global img_size, max_dist, scale, ratio

    img_1 = cv2.imread(p_lane_1[0][1])
    width, height = int(img_1.shape[1] * scale), int(img_1.shape[0] * scale)
    img_size = (height, width)
    ratio = img_size[1] / img_size[0]
    max_dist = math.sqrt(width ** 2 + height ** 2) / 2

    # ----------------------------------------------------------------------

    # image_dist = []
    bad_matches = []

    for file in os.listdir("./res/output/match"):
        if file.endswith(".jpg"):
            os.remove("./res/output/match/" + file)

    for file in os.listdir("./res/output/video"):
        if file.endswith(".jpg"):
            os.remove("./res/output/video/" + file)

    # p_lane_1 = deepcopy(p_lane_1)
    # p_list_2 = deepcopy(p_lane_2)

    match_list = []

    last_j = 0

    for i in range(0, len(p_lane_1)):

        avg_dist = None

        for j in range(0, len(p_lane_2)):

            # Comprobamos que la distancia entre coordenadas no sea excesiva
            j_distance = geopy.distance.geodesic(p_lane_1[i][2:4], p_lane_2[j][2:4]).m
            if j_distance > max_distance:
                continue

            # Calculamos la distancia para el elemento j
            avg_dist = sift_function(p_lane_1[i][1], p_lane_2[j][1])

            sys.stdout.write("\r- %s (%.2f %%) | %s (%.2f %%)" % (
                p_lane_1[i][0].split("_")[5], 100 * (i + 1) / len(p_lane_1),
                p_lane_2[j][0].split("_")[5], 100 * (j + 1) / len(p_lane_2)))

            # Si la distancia no es nula, sale del bucle
            if avg_dist is not None:
                last_j = j
                break

        # Si la distancia no es nula, añade la distancia a la lista y elimina el elemento j-1 utilizado (si j > 0).
        # Esto nos permite volver a utilizar j si no encontramos coincidencia con la imagen siguiente
        if avg_dist is not None:
            match_list.append(p_lane_1[i])
            match_list[-1].extend(p_lane_2[last_j])
            match_list[-1].append(avg_dist)
            match_list[-1].append("MATCHED")
            continue

        # Si sigue siendo nulo, pone la imagen en la lista mala
        if avg_dist is None:
            bad_matches.append(i)

    # La lista mala se rellena con valores vacíos para L2 y distancia nula
    for i in bad_matches:
        match_list.append(p_lane_1[i])
        match_list[-1].extend([""] * len(p_lane_2[last_j]))
        match_list[-1].append(None)
        match_list[-1].append("DEDUCTED")

    match_list.sort(key=lambda x: x[1])

    # Serie ascendente de match_list: acoplar imágenes de L2 consecutivas
    for i in range(1, len(match_list)):
        if match_list[i][4] == "" and match_list[i - 1][4] != "":

            x1 = match_list[i - 1][5].rfind("/") + 1
            x2 = match_list[i - 1][5].rfind(".")

            j = int(match_list[i - 1][5][int(x1): int(x2)])
            j = "./res/input/prepared_lane_2/" + str(j + 1) + ".jpg"

            if not os.path.isfile(j):
                print("no existe el fichero...")
                continue

            for k in range(0, len(p_lane_2)):
                if j in p_lane_2[k][1]:
                    match_list[i][4:8] = p_lane_2[k]
                    break

    # Serie descendente de match_list: acoplar imágenes de L2 consecutivas
    for i in range(len(match_list) - 1, -1, -1):
        if match_list[i][4] == "" and match_list[i + 1][4] != "":

            x1 = match_list[i + 1][5].rfind("/") + 1
            x2 = match_list[i + 1][5].rfind(".")

            j = int(match_list[i + 1][5][int(x1): int(x2)])
            j = "./res/input/prepared_lane_2/" + str(j - 1) + ".jpg"

            if not os.path.isfile(j):
                print("no existe el fichero...")
                continue

            for k in range(0, len(p_lane_2)):
                if j in p_lane_2[k][1]:
                    match_list[i][4:8] = p_lane_2[k]
                    break

    df = pd.DataFrame(match_list, columns=["1_S", "1_P", "1_LO", "1_LA", "2_S", "2_P", "2_LO", "2_LA", "DIST", "MATCH"])

    # Interpolación (para elemento de la lista mala)
    df = df.interpolate()  # interpolación lineal hacia adelante
    df = df.interpolate(limit=100, limit_direction="backward")  # interpolación lineal hacia atrás

    df.to_csv("./res/output/data/match_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".csv", sep=";", decimal=",",
              index=False)

    for index, row in df.iterrows():
        draw_overlap_transversal(row, str(index).zfill(len(str(len(df) - 1))))


def overlap_transversal(p_folder_l1, p_folder_l2, p_reverse):
    lane_1, lane_2 = prepare_data(p_folder_l1, p_folder_l2, p_reverse)
    match_images(lane_1, lane_2)


def overlap_longitudinal(p_folder_l1: str):
    """
    Calcula el solape longitudinal entre las imágenes de un carril
    :param p_folder_l1: directorio de las imágenes
    :return:
    """
    # Variables
    ls_image = []  # lista de imágenes
    sh_image = None  # tamaño de imagen
    ls_dist = []  # lista de distancias calculadas
    ls_link = []  # relación de fichero y distancia de recorte
    min_kp = 10  # numero mínimo de puntos clave en solape

    # Lista todos los ficheros JPEG de la carpeta
    for file in os.listdir(p_folder_l1):
        if file.endswith(".jpg"):
            ls_image.append(p_folder_l1 + "/" + file)

    # Recorre todas las imágenes hasta la penúltima
    for i in range(0, len(ls_image) - 1):

        img_1 = cv2.imread(ls_image[i])
        img_1 = cv2.rotate(img_1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_2 = cv2.imread(ls_image[i + 1])
        img_2 = cv2.rotate(img_2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ibw_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        ibw_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        sh_image = ibw_1.shape if sh_image is None else sh_image

        # -- SIFT --

        # Inicialización de SIFT
        sift = cv2.SIFT_create()

        # Puntos clave y descriptores
        kp1, des1 = sift.detectAndCompute(ibw_1, None)
        kp2, des2 = sift.detectAndCompute(ibw_2, None)

        # BFMatcher con parámetros por defecto
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Radio test
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(ibw_1, kp1, ibw_2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv2.resize(img3, (350, 800))
        img3 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow("", img3)
        cv2.waitKey(1)

        # Cálculo de la distancia de solape
        matches = np.asarray(good)
        if len(matches[:]) >= min_kp:  # si hay un mínimo de puntos clave
            avg_distance = get_distance(good, kp1, kp2, p_max=200, p_type="leg_x")

            if avg_distance is not None:  # si la distancia no es nula
                ls_dist.append(int(np.average(avg_distance)))

                sys.stdout.write("\r- Good match: %s, %d" % (ls_image[i].split("/")[-1], ls_dist[-1]))
                continue

        sys.stdout.write("\r- Bad match: %s" % ls_image[i].split("/")[-1])
        ls_dist.append(None)

    sys.stdout.write("\r")

    # Por cada elemento de la lista de distancia
    for i in range(0, len(ls_dist)):

        i_prev = None
        i_next = None

        # Si el elemento es nulo
        if ls_dist[i] is None:

            # Si el elemento no está en los extremos
            if 0 < i < len(ls_dist) - 1:
                for j in range(i - 1, -1, -1):  # Primer elemento anterior
                    if ls_dist[j] is not None:
                        i_prev = j
                        break

                for j in range(i + 1, len(ls_dist)):  # Primer elemento siguiente
                    if ls_dist[j] is not None:
                        i_next = j
                        break

            # Si es el primer elemento
            elif i == 0:
                for j in range(i + 1, len(ls_dist)):  # Primer elemento siguiente
                    if ls_dist[j] is not None:
                        i_next = j
                        break

            # Si es el último elemento
            elif i == len(ls_dist) - 1:
                for j in range(i - 1, -1, -1):  # Primer elemento anterior
                    if ls_dist[j] is not None:
                        i_prev = j
                        break

            # Si hay un valor anterior y posterior
            if i_prev is not None and i_next is not None:
                ls_dist[i] = int((ls_dist[i_prev] + ls_dist[i_next]) / 2)

            # Si hay un valor anterior
            elif i_prev is not None:
                ls_dist[i] = ls_dist[i_prev]

            # Si hay un valor posterior
            elif i_next is not None:
                ls_dist[i] = ls_dist[i_next]

        # Si es nulo, lo convierte en cero
        ls_dist[i] = 0 if ls_dist[i] is None else ls_dist[i]

        # Modifica el nombre de fichero de la imagen
        path_crop = ls_image[i].replace("param1", "cl-" + str(ls_dist[i]))

        # Lista para reemplazo de nombre de la imagen
        ls_link.append([ls_image[i], path_crop])

    for i in ls_link:
        print("-", i)


def overlap(folder_l1: str = "", folder_l2: str = ""):
    # overlap_longitudinal(folder_l1)
    overlap_transversal(folder_l1, folder_l2, 2)


if __name__ == "__main__":
    overlap("./res/input/lane_1", "./res/input/lane_2")
    # lane_1, lane_2 = prepare_data(2)
    #
    # overlap_transversal()
    #
    # overlap_longitudinal()
    #
    # match_images(lane_1, lane_2)

    # images = match_coordinates(lane_1, lane_2)
    #
    # df_img = pd.DataFrame(images, columns=["lane_1_src", "lane_1_prep", "lane_1_lat", "lane_1_lon",
    #                                        "lane_2_src", "lane_2_prep", "lane_2_lat", "lane_2_lon", "dist"])
    #
    # df_img.to_csv("./res/output/data/geomatch.csv", sep=";", decimal=",", index=False)
    #
    # fig = plt.figure()
    #
    # for idx in range(0, len(images) - 1):
    #     x1, x2 = images[idx][3], images[idx][7]
    #     y1, y2 = images[idx][2], images[idx][6]
    #     plt.plot([x1, x2], [y1, y2], 'r')
    #
    # x_list = np.concatenate((np.array(lane_1)[:, 3], np.array(lane_2)[:, 3]), axis=0).astype(float)
    # y_list = np.concatenate((np.array(lane_1)[:, 2], np.array(lane_2)[:, 2]), axis=0).astype(float)
    #
    # plt.scatter(x_list, y_list, s=1)
    #
    # plt.xlim([np.min(x_list), np.max(x_list)])
    # plt.ylim([np.min(y_list), np.max(y_list)])
    #
    # plt.axis("equal")
    # fig.savefig('./res/output/data/geomatch.svg')
