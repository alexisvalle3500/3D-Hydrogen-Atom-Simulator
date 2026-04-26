import glfw
from OpenGL.GL import *
import numpy as np
import time

# window dimensions
WIDTH, HEIGHT = 1440, 1080

# camera navigation state
orbit_yaw   = 0.0
orbit_pitch = 0.0
orbit_dist  = 14

# camera pan offset
pan_x = 0.0
pan_y = 0.0

def perspective(fov_deg, aspect, near, far):
    # build a perspective projection matrix from scratch
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    return np.array([
        [f / aspect, 0,  0,                               0],
        [0,          f,  0,                               0],
        [0,          0,  (far + near) / (near - far),    -1],
        [0,          0,  (2 * far * near) / (near - far), 0],
    ], dtype=np.float32).T

def look_at_modelview():
    # convert degrees to radians
    yaw   = np.radians(orbit_yaw)
    pitch = np.radians(orbit_pitch)

    # rotation around y axis
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw), 0],
        [0,            1, 0,           0],
        [-np.sin(yaw), 0, np.cos(yaw), 0],
        [0,            0, 0,           1],
    ], dtype=np.float32)

    # rotation around x axis
    Rx = np.array([
        [1, 0,              0,             0],
        [0, np.cos(pitch), -np.sin(pitch), 0],
        [0, np.sin(pitch),  np.cos(pitch), 0],
        [0, 0,              0,             1],
    ], dtype=np.float32)

    # translation back by orbit distance and pan
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = -pan_x
    T[1, 3] = -pan_y
    T[2, 3] = -orbit_dist

    return T @ Rx @ Ry

def main():
    glfw.init()

    # request opengl 3.3 core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WIDTH, HEIGHT, "hydrogen atom simulator", None, None)
    glfw.make_context_current(window)

    # enable depth testing so closer objects appear in front
    glEnable(GL_DEPTH_TEST)

    # enable blending for transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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