import math
import os
import shutil
import sys
# import time

import cv2
import geopy.distance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from PIL import Image, ImageFont, ImageDraw

# +info
# https://www.gpsworld.com/what-exactly-is-gps-nmea-data/
# https://www.earthpoint.us/convert.aspx


# Distancia máxima en metros: diagonal de la imagen
# 0.2098: relación cm/px calculada para imágenes corregidas de cámara 0
max_distance = 1.5 * 0.2098 * math.sqrt(2448 ** 2 + 968 ** 2) / 100

show_img = True
img_size = [0] * 2
ratio = 1
distance_type = "leg"
distance_operation = "avg"
scale = 0.5
max_dist = 0

min_keypoints = 5
image_group = 10


def fix_lane_coord(p_lane: list) -> list:
    """
    Corrige coordenadas repetidas consecutivas

    :param p_lane:
    :return:
    """
    # Si las coordenadas están repetidas, las corregimos
    for k in range(1, len(p_lane) - 2):
        lat_0, lat_1, lat_2 = p_lane[k - 1][-2], p_lane[k][-2], p_lane[k + 1][-2]  # latitud
        lon_0, lon_1, lon_2 = p_lane[k - 1][-1], p_lane[k][-1], p_lane[k + 1][-1]  # longitud

        if lat_1 == lat_0 and lon_1 == lon_0:
            p_lane[k][-2] = abs(lat_0 - lat_2) / 2 + min([lat_0, lat_2])
            p_lane[k][-1] = abs(lon_0 - lon_2) / 2 + min([lon_0, lon_2])

    return p_lane


def prepare_data(inverted: int = 0):
    """
    Prepara la información a partir de dos carpetas de entrada (L1 y L2)

    :param inverted: 0 (no lane reversed), 1 (lane 1), 2 (lane 2), 3 (both lanes)
    :return:
    """
    path_lane_1 = "./res/input/lane_1"
    path_lane_2 = "./res/input/lane_2"

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
    for file in os.listdir(path_lane_1):
        if file.endswith(".jpg"):
            # file = file.split("_")[4]
            list_lane_1.append(path_lane_1 + "/" + file)

    for file in os.listdir(path_lane_2):
        if file.endswith(".jpg"):
            # file = file.split("_")[4]
            list_lane_2.append(path_lane_2 + "/" + file)

    # Número máximo de caracteres en texto
    chr_max = len(str(max([len(list_lane_1), len(list_lane_2)])))

    # Carril 1
    for p_idx, element in enumerate(list_lane_1):
        # Asigna un nombre numérico a la imagen, e invierte el orden si el carril está invertido
        temp_name = str(len(list_lane_1) - p_idx - 1).zfill(chr_max) if inverted in [1, 3] else str(p_idx).zfill(
            chr_max)
        list_lane_1[p_idx] = [element, path_prepared_1 + "/" + temp_name + ".jpg"]

        # Copia la imagen y la gira si está invertida
        if inverted in [1, 3]:
            cv2.imwrite(list_lane_1[p_idx][1], cv2.rotate(cv2.imread(list_lane_1[p_idx][0]), cv2.ROTATE_180))
        else:
            shutil.copy(list_lane_1[p_idx][0], list_lane_1[p_idx][1])

        sys.stdout.write("\r- Preparing lane 1 (%.2f %%)" % (100 * (p_idx + 1) / len(list_lane_1)))

    print("")

    # Carril 2
    for p_idx, element in enumerate(list_lane_2):
        # Asigna un nombre numérico a la imagen, e invierte el orden si el carril está invertido
        temp_name = str(len(list_lane_2) - p_idx - 1).zfill(chr_max) if inverted in [2, 3] else str(p_idx).zfill(
            chr_max)
        list_lane_2[p_idx] = [element, path_prepared_2 + "/" + temp_name + ".jpg"]

        # Copia la imagen y la gira si está invertida
        if inverted in [2, 3]:
            cv2.imwrite(list_lane_2[p_idx][1], cv2.rotate(cv2.imread(list_lane_2[p_idx][0]), cv2.ROTATE_180))
        else:
            shutil.copy(list_lane_2[p_idx][0], list_lane_2[p_idx][1])

        sys.stdout.write("\r- Preparing lane 2 (%.2f %%)" % (100 * (p_idx + 1) / len(list_lane_2)))

    print("")

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


def get_distance(p_good, kp1, kp2):
    for k, element in enumerate(p_good):
        p_good[k] = element[0]

    # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python

    list_kp1 = []
    list_kp2 = []

    for x in p_good:
        img1_idx = x.queryIdx
        img2_idx = x.trainIdx

        (x_1, y_1) = kp1[img1_idx].pt
        (x_2, y_2) = kp2[img2_idx].pt

        list_kp1.append((x_1, y_1))
        list_kp2.append((x_2 + img_size[1], y_2))

    dist = []

    for k in range(0, len(list_kp1)):
        x = np.abs(list_kp2[k][0] - list_kp1[k][0])
        y = np.abs(list_kp2[k][1] - list_kp1[k][1])

        # Opciones para el cálculo de la distancia: "hypotenuse", "leg", else
        if distance_type == "hypotenuse":
            # Teorema de Pitágoras
            distance = np.sqrt(x ** 2 + y ** 2)

        elif distance_type == "leg":
            # Cateto x
            distance = x
        else:
            # Promedio entre hipotenusa y cateto x
            distance = np.average([np.sqrt(x ** 2 + y ** 2), x])

        if distance <= max_dist:
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


def draw_knn(_p_i1, _p_i2, p_i3, p_f1, p_f2, p_avg_dist, p_match, ):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    cv2.putText(p_i3, "L1-%s | L2-%s" % (p_f1, p_f2), (30, 60), font, font_scale, font_color, thickness,
                line_type)

    text = "%.0f px LIM | %.0f px AVG" % (
        (max_dist / scale), (p_avg_dist / scale) if p_avg_dist is not None else -1)

    cv2.putText(p_i3, text, (int(150 * ratio) + 30, 30), font, font_scale, font_color, thickness,
                line_type)

    cv2.putText(p_i3, "Min. kp: %d | Keypoints: %d" % (min_keypoints, p_match),
                (int(150 * ratio) + 30, 60), font, font_scale, font_color, thickness, line_type)

    cv2.imwrite("./out/" + p_f1 + "-" + p_f2 + "_" + str(int(p_avg_dist / scale)) + ".jpg", p_i3)

    # cropped_image = np.concatenate((p_i1[0:-int(p_avg_dist / scale), :], p_i2), axis=1)


def sift_function(p_lane_1, p_lane_2):
    """
    
    :param p_lane_1: 
    :param p_lane_2:
    :return: 
    """

    avg_dist = None

    try:
        img_1 = cv2.imread(p_lane_1)
        img_2 = cv2.imread(p_lane_2)

        img1 = cv2.resize(img_1, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img_2, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        f1 = p_lane_1[-7:-4]
        f2 = p_lane_2[-7:-4]

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
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img3 = cv2.resize(img3, (int(ratio * 2 * 80), 80), interpolation=cv2.INTER_AREA)

        cv2.imshow("", img3)
        cv2.waitKey(1)

        # Distances
        matches = np.asarray(good)
        if len(matches[:]) >= min_keypoints:
            avg_distance = get_distance(good, kp1, kp2)

            if avg_distance is not None:
                avg_dist = int(np.average(avg_distance))

                draw_knn(img1, img2, img3, f1, f2, avg_distance, len(matches[:]))

        return avg_dist

    except Exception as e:
        print(e)
        return avg_dist


def match_images():
    p_folder1 = "./res/input/prepared_lane_1"
    p_folder2 = "./res/input/prepared_lane_2"

    list_lane_1 = []
    list_lane_2 = []

    image_dist = []
    # bad_matches = []

    for file in os.listdir("./out"):
        if file.endswith(".jpg"):
            os.remove("./out" + "/" + file)

    for file in os.listdir(p_folder1):
        if file.endswith(".jpg"):
            list_lane_1.append(p_folder1 + "/" + file)

    for file in os.listdir(p_folder2):
        if file.endswith(".jpg"):
            list_lane_2.append(p_folder2 + "/" + file)

    # start_time = time.time()
    # count = 0

    global img_size, max_dist, scale, ratio

    img_1 = cv2.imread(list_lane_1[0])
    width, height = int(img_1.shape[1] * scale), int(img_1.shape[0] * scale)
    img_size = (height, width)
    ratio = img_size[1] / img_size[0]

    max_dist = math.sqrt(width ** 2 + height ** 2) / 2

    last_j = 0

    for i in range(0, len(list_lane_1)):

        avg_dist = sift_function(list_lane_1[i], list_lane_2[last_j])

        sys.stdout.write("\r- %.2f | - %d" % (100 * (i + 1) / len(list_lane_1), last_j))

        if avg_dist is None:
            for j in range(0, len(list_lane_2)):
                avg_dist = sift_function(list_lane_1[i], list_lane_2[j])

                sys.stdout.write("\r- %.2f | %.2f" % (100 * (i + 1) / len(list_lane_1),
                                                      100 * (j + 1) / len(list_lane_2)))

                if avg_dist is not None:
                    last_j = j
                    break

        if avg_dist is not None:
            image_dist.append(avg_dist)
            del list_lane_2[last_j]
            continue

    for k in image_dist:
        print(k)


if __name__ == "__main__":
    lane_1, lane_2 = prepare_data(2)

    lane_1 = append_coordinates(lane_1)
    lane_2 = append_coordinates(lane_2)

    images = match_coordinates(lane_1, lane_2)

    df_img = pd.DataFrame(images, columns=["lane_1_src", "lane_1_prep", "lane_1_lat", "lane_1_lon",
                                           "lane_2_src", "lane_2_prep", "lane_2_lat", "lane_2_lon", "dist"])

    # df_img.to_csv("./geomatch.csv", sep=";", decimal=",", index=False)

    fig = plt.figure()

    for idx in range(0, len(images) - 1):
        x1, x2 = images[idx][3], images[idx][7]
        y1, y2 = images[idx][2], images[idx][6]
        plt.plot([x1, x2], [y1, y2], 'r')

    x_list = np.concatenate((np.array(lane_1)[:, 3], np.array(lane_2)[:, 3]), axis=0).astype(float)
    y_list = np.concatenate((np.array(lane_1)[:, 2], np.array(lane_2)[:, 2]), axis=0).astype(float)

    plt.scatter(x_list, y_list, s=1)

    plt.xlim([np.min(x_list), np.max(x_list)])
    plt.ylim([np.min(y_list), np.max(y_list)])

    plt.axis("equal")
    fig.savefig('./geomatch.svg')

    match_images()
