import math
import os
import shutil

import cv2
import geopy.distance

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

    return list_lane_1, list_lane_2


def transform_coordinates(p_coordinate: str):
    """Dec Mins 	3918.67633N10217.50783W
                    ddmm.mmmm dddmm.mmmm"""

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

        p_lane[idx].append([lat, lon])

    return p_lane


if __name__ == "__main__":
    lane_1, lane_2 = prepare_data(0)

    lane_1 = append_coordinates(lane_1)
    lane_2 = append_coordinates(lane_2)

    array_dist = []

    for idx_i, lane_i in enumerate(lane_1):
        array_dist_i = []

        nearest = [None, None]
        i_position = lane_i[2]

        for lane_j in lane_2:
            j_position = lane_j[2]
            j_distance = geopy.distance.geodesic(i_position, j_position).m

            if nearest[0] is None or j_distance < nearest[0]:
                nearest[0] = j_distance
                nearest[1] = lane_j

        if nearest[0] < max_distance:
            array_dist.append([lane_i, nearest[1], nearest[0]])

    if len(array_dist) > 0:

        for unit in array_dist:
            print(unit[0][2][1], unit[0][2][0],  # longitud, latitud (L1)
                  unit[1][2][1], unit[1][2][0],  # longitud, latitud (L2)
                  unit[2])  # distancia
