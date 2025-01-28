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
    margin = 224 * 0.12
    window_height = 224 * 3 + offset
    window_width =  int(224 * 5 + 2 * margin)
    init_window(window_width, window_height, "Veggie, Search!")
    order = None
    similarity = None
    models = [ model_name for model_name in features.models]
    print(models)
    selected_model = 0

    scroll_y = 0
    scroll_speed = 224/3
    query_img_path = ""

    background_color = Color(10,10,10,255)

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
        elif is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
            if get_mouse_position().y < (offset - scroll_y):
                selected_model= (selected_model+1) % len(models)
                change_order = True
        if change_order:
            order,similarity = features.get_image_order(image_path, models[selected_model])
            change_order = False
            scroll_y = 0
        scroll_y -=get_mouse_wheel_move()*scroll_speed
        if scroll_y < 0:
            scroll_y = 0

        begin_drawing()
        clear_background(background_color)
        if order:
            draw_text("Model: " + models[selected_model], 10, 10-int(scroll_y) , 32, LIGHTGRAY);
            top_row = max(0, int((scroll_y-offset)//224))
            special_texture = None
            special_pos = None
            special_name = None
            special_score = None
            for i in range(5*5):
                name = features.model_imagenames[models[selected_model]][order[i+top_row*5]]
                t = get_texture(name)
                x_pos = (i%5)*224+margin
                y_pos = (i//5) * 224 + 224 * top_row + offset - int(scroll_y)
                pos = Vector2(x_pos, y_pos)
                mouse_pos = get_mouse_position()
                if  y_pos < mouse_pos.y and mouse_pos.y < y_pos + 224 and x_pos < mouse_pos.x and mouse_pos.x < x_pos + 224:
                    special_texture = t
                    special_pos = Vector2(x_pos - 224*0.1, y_pos - 224*0.1)
                    special_name = name
                    special_score = f"Similarity: {similarity[order[i+top_row*5]]*100:.2f}%"
                draw_texture_ex(t, pos, 0.0, 1.0, WHITE)
            if special_texture is not None:
                draw_texture_ex(special_texture, special_pos, 0.0, 1.2, WHITE)
                draw_rectangle(int(special_pos.x), int(special_pos.y + 224 * 1.2), int(224 * 1.2), 44, background_color)
                draw_text(special_score, int(special_pos.x) + 7, int(special_pos.y + 224 * 1.2) + 10, 20, WHITE)
                draw_text(name, int(special_pos.x) + 7, int(special_pos.y + 224 * 1.2) + 28, 12, WHITE)
        else:
            msg = "Drop image to perform search"
            draw_text(msg, int(window_width/2 - measure_text(msg, 48)/2), int(window_height/2-24), 48, LIGHTGRAY);
        end_drawing()
    for _, texture in textures.items():
        unload_texture(texture)
    close_window()

def get_texture(path):
    if path in textures:
        return textures[path]
    else:
        texture = load_texture(path)
        textures[path] = texture
        return texture

if __name__ == "__main__":
    main()
