import glfw

WIDTH, HEIGHT = 1440, 1080

def main():
    glfw.init()
    window = glfw.create_window(WIDTH, HEIGHT, "hydrogen atom simulator", None, None)
    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()