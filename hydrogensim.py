import glfw
from OpenGL.GL import *
import time

# window dimensions
WIDTH, HEIGHT = 1440, 1080

def main():
    glfw.init()

    # request opengl 3.3 core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WIDTH, HEIGHT, "hydrogen atom simulator", None, None)
    glfw.make_context_current(window)

    # track time between frames
    prev = time.time()

    while not glfw.window_should_close(window):
        now = time.time()
        dt = now - prev
        prev = now

        # clear to dark blue-black background
        glClearColor(0.03, 0.03, 0.07, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()