import sys
import features
from pyray import *
from PIL import Image
import numpy as np
from cffi import FFI


order = None
show_images = False
textures = {}
def main():
    features.load_features()
    offset = 60
    # print(features.models)
    window_height = 224 * 3 + offset
    window_width =  224 * 5
    # button_width = 120
    init_window(window_width, window_height, "Veggie Search")
    order = None
    models = [ model_name for model_name in features.models]
    print(models)
    selected_model = 0

    scroll_y = 0
    scroll_speed = 224/3
    query_img_path = ""

    change_order = False
    while not window_should_close():

        if is_file_dropped():
            files = load_dropped_files()
            image_path = ffi.string(files.paths[0]).decode('utf-8')
            unload_dropped_files(files)
            query_img_path = image_path
            change_order = True
        if is_key_pressed(KEY_SPACE):
            selected_model= (selected_model+1) % len(models)
            change_order = True
        if change_order:
            order = features.get_image_order(image_path, models[selected_model])
            change_order = False
            scroll_y = 0
        scroll_y -=get_mouse_wheel_move()*scroll_speed
        if scroll_y < 0:
            scroll_y = 0

        begin_drawing()
        clear_background(BLACK)
        if order:
            # for i, model in enumerate(models):
                # x_pos = i * (button_width + 20) + 20  # Add some spacing
                # print(x_pos)
                # draw_rectangle(x_pos, 20, button_width, 60,  DARKGRAY)
                # draw_rectangle_lines(x_pos, 20, button_width, 60, DARKGREEN);
                # DrawText( processText[i], (int)( toggleRecs[i].x + toggleRecs[i].width/2 - MeasureText(processText[i], 10)/2), (int) toggleRecs[i].y + 11, 10, ((i == currentProcess) || (i == mouseHoverRec)) ? DARKBLUE : DARKGRAY);
            draw_text("Model: " + models[selected_model], 10, 10-int(scroll_y) , 32, LIGHTGRAY);
            top_row = max(0, int((scroll_y-offset)//224))
            for i in range(5*5):
                draw_texture(get_texture(features.model_imagenames[models[selected_model]][order[i+top_row*5]]), (i%5)*224, (i//5) * 224 + 224 * top_row + offset - int(scroll_y), WHITE)
        else:
            msg = "Drop image to perform search"
            draw_text(msg, int(window_width/2 - measure_text(msg, 48)/2), int(window_height/2-24), 48, LIGHTGRAY);
        end_drawing()
    for _, texture in textures.items():
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
