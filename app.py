# import imgui
# from imgui.integrations.glfw import GlfwRenderer
# import OpenGL.GL as gl
# import glfw
import features

def main():
    features.load_features()


if __name__=="__main__":
    main()











# imgui.create_context()
# 
# window = impl_glfw_init()
# 
# impl = GlfwRenderer(window)
# while not glfw.window_should_close(window):
#     glfw.poll_events()
#     impl.process_inputs()
#     imgui.new_frame()
#     imgui.text("Bar")
#     gl.glClearColor(1.0,1.0,1.0,1)
#     gl.glClear(gl.GL_COLOR_BUFFER_BIT)
#     imgui.render()
#     impl.render(imgui.get_draw_data())
#     glfw.swap_buffers(window)
# imp.shutdown()
# glfw.terminate()
