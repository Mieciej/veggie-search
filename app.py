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


def main():
    # features.load_features()
    init_window(800, 450, "Hello")
    while not window_should_close():
        if is_file_dropped():
            files = load_dropped_files()
            print(ffi.string(files.paths[0]).decode('utf-8'))
            unload_dropped_files(files)
        begin_drawing()
        clear_background(WHITE)
        draw_text("Hello world", 190, 200, 20, VIOLET)
        end_drawing()
    close_window()



def drop_callback(window, paths):
    image_path = paths[0]
    order = features.get_image_order(image_path, "MACIEK1.keras")






if __name__ == "__main__":
    main()
