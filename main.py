import math
import os
import shutil
import sys

import cv2
import geopy.distance
# import pandas as pd
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# +info
# https://www.gpsworld.com/what-exactly-is-gps-nmea-data/
# https://www.earthpoint.us/convert.aspx


# Distancia máxima en metros: diagonal de la imagen
# 0.2098: relación cm/px calculada para imágenes corregidas de cámara 0
max_distance = 0.2098 * math.sqrt(2448 ** 2 + 968 ** 2) / 100

show_img = True
img_size = [0] * 2
distance_type = "hypotenuse"
distance_operation = "avg"
scale = 1
max_dist = 200 * scale
min_keypoints = 10
image_group = 10


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
    for p_idx, element in enumerate(p_lane):
        name = element[0].split("/")[-1]
        lat, lon = transform_coordinates(" ".join([name.split("_")[5], name.split("_")[6]]))

        p_lane[p_idx].extend([lat, lon])

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

    return p_array  # pd.DataFrame(p_array, columns=["lane_1_src", "lane_1_prep", "lane_1_lat",
    # "lane_1_lon", "lane_2_src", "lane_2_prep", "lane_2_lat", "lane_2_lon", "dist"])


def get_distance(p_good):
    for i, element in enumerate(p_good):
        p_good[i] = element[0]

    # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python

    list_kp1 = []
    list_kp2 = []

    for x in p_good:
        img1_idx = x.queryIdx
        img2_idx = x.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        list_kp1.append((x1, y1))
        list_kp2.append((x2 + img_size[1], y2))

    dist = []

    for i in range(0, len(list_kp1)):
        x = np.abs(list_kp2[i][0] - list_kp1[i][0])
        y = np.abs(list_kp2[i][1] - list_kp1[i][1])

        if distance_type == "hypotenuse":
            distance = np.sqrt(x ** 2 + y ** 2)

        elif distance_type == "leg":
            distance = x
        else:
            distance = np.average([np.sqrt(x ** 2 + y ** 2), x])

        if distance <= max_dist:
            dist.append(distance)
        # print(list_kp1[i], " -> ", list_kp2[i], " -> ", np.sqrt(x ** 2 + y ** 2))

    if len(dist) > 0:
        if distance_operation == "avg":
            return np.average(dist) / scale
        elif distance_operation == "max":
            return np.max(dist) / scale
        elif distance_operation == "min":
            return np.min(dist) / scale

    return None


if __name__ == "__main__":
    lane_1, lane_2 = prepare_data(2)

    lane_1 = append_coordinates(lane_1)
    lane_2 = append_coordinates(lane_2)

    images = match_coordinates(lane_1, lane_2)

    image_dist = []
    bad_matches = []

    for idx in range(0, len(images)):

        bad_matches.append(idx)

        img_1 = cv2.imread(images[idx][1])

        if images[idx][5] is not None:
            img_2 = cv2.imread(images[idx][5])
        else:
            continue

        img1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        if img_size[0] == 0:
            img_size = img1.shape
        img3 = None

        try:
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
            img3 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)

            # Distances
            matches = np.asarray(good)
            if len(matches[:]) >= min_keypoints:
                avg_distance = get_distance(good)
                if avg_distance is not None:
                    image_dist.append(int(np.average(avg_distance)))

                else:
                    print("(%.2f %%) %s - %s - Not enough points under max distance" % (
                        100 * (idx + 1) / len(images), idx, images[idx][1]))
                    continue
            else:
                print(
                    '(%.2f %%) %s - %s - Can’t find enough keypoints.' % (
                    100 * (idx + 1) / len(images), images[idx][2], images[idx][5]))
                continue
        except Exception as e:
            print(e)
            continue

        overlap = int(image_dist[-1] * scale)

        # Distance
        img = Image.fromarray(img3)
        font = ImageFont.truetype("Roboto-Regular.ttf", int(img_size[0] / 12))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Mean distance (re): %d px" % image_dist[-1], (0, 255, 255), font=font)
        if scale != 1:
            draw.text((10, 100), "Mean distance (sc): %d px" % overlap, (0, 0, 0), font=font)
        img.save("res/output/dist/%s_%s_%s.jpg" % (str(idx + 1).zfill(4), str(idx + 2).zfill(4), str(overlap)))

        # Concatenation
        img = np.concatenate((img_1, img_2[0:img_size[0], image_dist[-1]:img_size[1]]), axis=1)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("res/output/join/%s_%s_%s.jpg" % (str(idx + 1).zfill(4), str(idx + 2).zfill(4), str(overlap)),
                    img)

        print("(%.2f %%) %s - %s - %d px - %d px" % (
            100 * (idx + 1) / len(images), idx, idx + 1, overlap, np.average(image_dist) * scale))

        # If the match is ok, remove the pair from the list
        bad_matches.pop()
