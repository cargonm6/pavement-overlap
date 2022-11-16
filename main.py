import math
import os
import shutil
import sys
import time

import cv2
import geopy.distance
# import pandas as pd
import numpy as np

# from PIL import Image, ImageFont, ImageDraw

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
scale = 0.1
max_dist = 0

min_keypoints = 0
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


def get_distance(p_good, kp1, kp2):
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

    start_time = time.time()
    count = 0

    global img_size, max_dist, scale

    img_1 = cv2.imread(list_lane_1[0])

    width = int(img_1.shape[1] * scale)
    height = int(img_1.shape[0] * scale)

    img_size = (height, width)
    ratio = img_1.shape[1] / img_1.shape[0]

    max_dist = 200000000 * math.sqrt(width ** 2 + height ** 2) / 2

    # init_j = 0

    for i in range(10, len(list_lane_1)):
        # print("-", len(list_lane_2) - init_j)

        print("\n-", len(list_lane_2))

        for j in range(0, len(list_lane_2)):

            try:
                img_1 = cv2.imread(list_lane_1[i])
                img_2 = cv2.imread(list_lane_2[j])

            except Exception as e:
                print(e)
                input(".~-")
                break

            img1 = cv2.resize(img_1, (width, height), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img_2, (width, height), interpolation=cv2.INTER_AREA)

            # cv2.imshow("", img_1)
            # cv2.waitKey(0)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # img1 = img_filter(img1, "gamma")
            # img2 = img_filter(img2, "gamma")

            f1 = list_lane_1[i][-7:-4]
            f2 = list_lane_2[j][-7:-4]

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

                img3 = cv2.resize(img3, (int(300 * ratio), 150), interpolation=cv2.INTER_AREA)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                line_type = 2

                count += 1
                rate = count / (time.time() - start_time)

                text = "%.2f %% | %.2f %% | %.2f samples/s" % (
                    100 * (i + 1) / len(list_lane_1), 100 * (j + 1) / len(list_lane_2), rate)

                cv2.putText(img3, text, (30, 30), font, font_scale, font_color, thickness, line_type)
                cv2.putText(img3, "L1-%s | L2-%s" % (f1, f2), (30, 60), font, font_scale, font_color, thickness,
                            line_type)
                cv2.putText(img3, "Min. keypoints: %d" % min_keypoints, (int(150 * ratio) + 30, 60), font, font_scale,
                            font_color, thickness, line_type)

                # img3 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)

                # Distances
                matches = np.asarray(good)
                if len(matches[:]) >= min_keypoints:
                    avg_distance = get_distance(good, kp1, kp2)

                    text = "%.0f px LIM | %.0f px AVG" % (
                        max_dist / scale, avg_distance / scale if avg_distance is not None else -1)

                    cv2.putText(img3, text, (int(150 * ratio) + 30, 30), font, font_scale, font_color, thickness,
                                line_type)

                    cv2.imshow("", img3)
                    cv2.waitKey(1)
                    cv2.imwrite("./video/" + str(count).zfill(8) + ".jpg", img3)

                    if avg_distance is not None:
                        image_dist.append(
                            [list_lane_1[i], list_lane_2[j], int(np.average(avg_distance)), len(matches[:])])

                        sys.stdout.write(
                            "\r(%.2f %% | %.2f %%) Coincidence: %d points [%s-%s]. Speed: %.2f samples/sec\n" %
                            (100 * (i + 1) / len(list_lane_1), 100 * (j + 1) / len(list_lane_2),
                             len(matches[:]), f1, f2, rate))

                        cv2.imwrite("./out/" + f1 + "-" + f2 + ".jpg", img3)

                        del list_lane_2[j]

                        # init_j = j + 1

                        break

                    else:
                        sys.stdout.write(
                            "\r(%.2f %% | %.2f %%) Not enough points under max distance. Speed: %.2f samples/sec" % (
                                100 * (i + 1) / len(list_lane_1), 100 * (j + 1) / len(list_lane_2), rate))
                        continue
                else:
                    text = "%.0f px LIM | %.0f px AVG" % (max_dist / scale, -1)

                    cv2.putText(img3, text, (int(150 * ratio) + 30, 30), font, font_scale, font_color, thickness,
                                line_type)

                    cv2.imshow("", img3)
                    cv2.waitKey(1)

                    sys.stdout.write("\r(%.2f %% | %.2f %%) Can’t find enough keypoints. Speed: %.2f samples/sec" % (
                        100 * (i + 1) / len(list_lane_1), 100 * (j + 1) / len(list_lane_2), rate))

                    cv2.imwrite("./video/" + str(count).zfill(8) + ".jpg", img3)
                    continue
            except Exception as e:
                # print(e)
                continue

    for i in image_dist:
        print(i)


if __name__ == "__main__":
    # lane_1, lane_2 = prepare_data(2)
    #
    # lane_1 = append_coordinates(lane_1)
    # lane_2 = append_coordinates(lane_2)
    #
    # images = match_coordinates(lane_1, lane_2)
    #
    match_images()
