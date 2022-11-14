import math
import os
import shutil
import sys

import cv2
import geopy.distance
import pandas as pd

# +info
# https://www.gpsworld.com/what-exactly-is-gps-nmea-data/
# https://www.earthpoint.us/convert.aspx


# Distancia máxima en metros: diagonal de la imagen
# 0.2098: relación cm/px calculada para imágenes corregidas de cámara 0
max_distance = 0.2098 * math.sqrt(2448 ** 2 + 968 ** 2) / 100


def prepare_data(inverted: int = 0):
    """

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
    for idx, element in enumerate(list_lane_1):
        # Asigna un nombre numérico a la imagen, e invierte el orden si el carril está invertido
        temp_name = str(len(list_lane_1) - idx - 1).zfill(chr_max) if inverted in [1, 3] else str(idx).zfill(chr_max)
        list_lane_1[idx] = [element, path_prepared_1 + "/" + temp_name + ".jpg"]

        # Copia la imagen y la gira si está invertida
        if inverted in [1, 3]:
            cv2.imwrite(list_lane_1[idx][1], cv2.rotate(cv2.imread(list_lane_1[idx][0]), cv2.ROTATE_180))
        else:
            shutil.copy(list_lane_1[idx][0], list_lane_1[idx][1])

        sys.stdout.write("\r- Preparing lane 1 (%.2f %%)" % (100 * (idx + 1) / len(list_lane_1)))

    print("")

    # Carril 2
    for idx, element in enumerate(list_lane_2):
        # Asigna un nombre numérico a la imagen, e invierte el orden si el carril está invertido
        temp_name = str(len(list_lane_2) - idx - 1).zfill(chr_max) if inverted in [2, 3] else str(idx).zfill(chr_max)
        list_lane_2[idx] = [element, path_prepared_2 + "/" + temp_name + ".jpg"]

        # Copia la imagen y la gira si está invertida
        if inverted in [2, 3]:
            cv2.imwrite(list_lane_2[idx][1], cv2.rotate(cv2.imread(list_lane_2[idx][0]), cv2.ROTATE_180))
        else:
            shutil.copy(list_lane_2[idx][0], list_lane_2[idx][1])

        sys.stdout.write("\r- Preparing lane 2 (%.2f %%)" % (100 * (idx + 1) / len(list_lane_2)))

    print("")

    return list_lane_1, list_lane_2


def transform_coordinates(p_coordinate: str):
    """
    Dec Mins: ddmm.mmmmm(N/S) dddmm.mmmmm(E/W)
    """

    lat, lon = p_coordinate.split(" ")

    lat = [lat[0:2], lat[2:-1], lat[-1]]
    lon = [lon[0:3], lon[3:-1], lon[-1]]

    lat = (float(lat[0]) + float(lat[1]) / 60) * (-1 if lat[2] in ["S"] else 1)
    lon = (float(lon[0]) + float(lon[1]) / 60) * (-1 if lon[2] in ["W"] else 1)

    return lat, lon


def append_coordinates(p_lane):
    for idx, element in enumerate(p_lane):
        name = element[0].split("/")[-1]
        lat, lon = transform_coordinates(" ".join([name.split("_")[5], name.split("_")[6]]))

        p_lane[idx].extend([lat, lon])

    return p_lane


def match_coordinates(p_lane_1, p_lane_2):
    p_array = []
    bad_match = 0

    for idx_i, lane_i in enumerate(p_lane_1):

        nearest_dst = None
        nearest_img = None

        for lane_j in p_lane_2:
            j_distance = geopy.distance.geodesic(lane_i[2:4], lane_j[2:4]).m

            if nearest_dst is None or nearest_dst > j_distance:
                nearest_dst = j_distance
                nearest_img = lane_j

        p_array.append(lane_i)
        if nearest_dst is not None and nearest_dst < max_distance:
            p_array[-1].extend(nearest_img)
            p_array[-1].append(nearest_dst)
        else:
            p_array[-1].extend([None] * (len(lane_i) + 1))
            bad_match += 1

        sys.stdout.write("\r- Matching coordinates (%.2f %%): %d/%d (ok/ko)" % (
            100 * (idx_i + 1) / len(p_lane_1), idx_i + 1 - bad_match, bad_match))

    print("")

    return pd.DataFrame(p_array, columns=["lane_1_src", "lane_1_prep", "lane_1_lat", "lane_1_lon",
                                          "lane_2_src", "lane_2_prep", "lane_2_lat", "lane_2_lon", "dist"])


if __name__ == "__main__":
    lane_1, lane_2 = prepare_data(0)

    lane_1 = append_coordinates(lane_1)
    lane_2 = append_coordinates(lane_2)

    array_dist = match_coordinates(lane_1, lane_2)

    print(array_dist)
