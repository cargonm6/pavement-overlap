import os
import subprocess
import traceback


def create_video(p_file: str = "out.avi", p_rate: int = 12):
    """
    Crea un vídeo con FFMPEG a partir de un conjunto de imágenes
    :return:
    """

    if not os.path.isfile("./ffmpeg.exe"):
        print("No se encontró el archivo \"./ffmpeg.exe\"")
        return

    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir_path)
        os.remove("./" + p_file) if os.path.isfile("./" + p_file) else 0
        subprocess.call(["ffmpeg", "-r", str(p_rate), "-i", "img%3d.jpg", p_file])
    except Exception:
        print(traceback.format_exc())


if __name__ == "__main__":
    create_video()
