import os
import subprocess
import time
from zipfile import ZipFile

import requests as requests


def create_video(p_dir="", p_file: str = "out.avi", p_rate: int = 12):
    """
    Crea un vídeo con FFMPEG a partir de un conjunto de imágenes
    :return:
    """

    dir_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + p_dir

    # Elimina duplicado
    os.remove(dir_path + "/" + p_file) if os.path.isfile(dir_path + "/" + p_file) else 0

    # Cambia el directorio de trabajo
    # os.chdir(dir_path)

    # Si es AVI, valor por defecto
    if p_file[-3:] == "avi":
        subprocess.call(["ffmpeg", "-r", str(p_rate), "-i", f"{dir_path}/img%3d.jpg", f"{dir_path}/{p_file}"])

    # Si es MP4, libx264
    else:
        subprocess.call(
            ["ffmpeg", "-r", str(p_rate), "-i", f"{dir_path}/img%3d.jpg", "-c:v", "libx264", "-crf", "20",
             f"{dir_path}/{p_file}"])


if __name__ == "__main__":

    # Comprueba si está el programa FFMPEG
    if not os.path.isfile("ffmpeg.exe"):

        # Descargamos FFMPEG desde GitHub
        print("- Descargando FFMPEG...")
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        res = requests.get(url)
        open("ffmpeg.zip", "wb").write(res.content)

        # Extraemos el ejecutable en el directorio de trabajo
        print("- Extrayendo FFMPEG...")
        with ZipFile('ffmpeg.zip', 'r') as zipObj:
            listOfFileNames = zipObj.namelist()
            for fileName in listOfFileNames:
                if fileName.endswith('ffmpeg.exe'):
                    unpacked = open('ffmpeg.exe', 'wb')
                    unpacked.write(zipObj.read(fileName))
                    unpacked.close()

        print("- Eliminando archivos temporales...")
        # Eliminamos el archivo ZIP
        if os.path.isfile("ffmpeg.zip"):
            os.remove("ffmpeg.zip")

    print("- Generando vídeo de solape transversal...")
    create_video("/res/output/video", p_file="out.avi")

    print("- Generando vídeo de solape en perspectiva...")
    create_video("/res/output/video_perspective", p_file="out.mp4")
