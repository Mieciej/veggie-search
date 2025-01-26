import sys
import features
from pyray import *
from PIL import Image
import numpy as np
from cffi import FFI
WINDOW_HEIGHT = 1280
WINDOW_WIDTH = 720

order = None
show_images = False
textures = {}
def main():
    features.load_features()
    init_window(224*5, 224*3, "Hello")
    order = None
    while not window_should_close():
        if is_file_dropped():
            files = load_dropped_files()
            image_path = ffi.string(files.paths[0]).decode('utf-8')
            unload_dropped_files(files)
            order = features.get_image_order(image_path ,"MACIEK1.keras")
        begin_drawing()
        clear_background(WHITE)
        if order:
            for y in range(3):
                for x in range(5):
                    draw_texture(get_texture(features.model_imagenames["MACIEK1.keras"][order[x+y]]), x*224, y*224, WHITE)
        end_drawing()
    for _, texture in textures:
        unload_texture(texture)
    close_window()



def drop_callback(window, paths):
    image_path = paths[0]
    order = features.get_image_order(image_path, "MACIEK1.keras")

def get_texture(path):
    if path in textures:
        return textures[path]
    else:
        texture = load_texture(path)
        textures[path] = texture
        return texture


if __name__ == "__main__":
    main()
