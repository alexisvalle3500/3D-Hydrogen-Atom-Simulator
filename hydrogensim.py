import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import time

# window dimensions
WIDTH, HEIGHT = 1440, 1080

# physical constants in scaled units
A0   = 1.0
HBAR = 1.0
ME   = 1.0

# quantum state
quantum = {"N": 4, "L": 2, "M": 0}

# camera navigation state
orbit_yaw   = 0.0
orbit_pitch = 0.0
orbit_dist  = 14

# camera pan offset
pan_x = 0.0
pan_y = 0.0

# mouse state
last_mouse_x = 0.0
last_mouse_y = 0.0
mouse_left  = False
mouse_right = False

# basic passthrough vertex shader
VERTEX_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 vColor;

uniform mat4 uMVP;

void main() {
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# basic passthrough fragment shader
FRAGMENT_SRC = """
#version 330 core

in  vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
"""

def _compile_shader(src, kind):
    shader = glCreateShader(kind)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def build_program():
    vs = _compile_shader(VERTEX_SRC,  GL_VERTEX_SHADER)
    fs = _compile_shader(FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    for s in (vs, fs):
        glAttachShader(prog, s)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    for s in (vs, fs):
        glDeleteShader(s)
    return prog

def perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    return np.array([
        [f / aspect, 0,  0,                               0],
        [0,          f,  0,                               0],
        [0,          0,  (far + near) / (near - far),    -1],
        [0,          0,  (2 * far * near) / (near - far), 0],
    ], dtype=np.float32).T

def look_at_modelview():
    yaw   = np.radians(orbit_yaw)
    pitch = np.radians(orbit_pitch)

    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw), 0],
        [0,            1, 0,           0],
        [-np.sin(yaw), 0, np.cos(yaw), 0],
        [0,            0, 0,           1],
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0,              0,             0],
        [0, np.cos(pitch), -np.sin(pitch), 0],
        [0, np.sin(pitch),  np.cos(pitch), 0],
        [0, 0,              0,             1],
    ], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[0, 3] = -pan_x
    T[1, 3] = -pan_y
    T[2, 3] = -orbit_dist

    return T @ Rx @ Ry

def mouse_button_callback(window, button, action, mods):
    global mouse_left, mouse_right, last_mouse_x, last_mouse_y
    x, y = glfw.get_cursor_pos(window)
    last_mouse_x, last_mouse_y = x, y
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_left = action == glfw.PRESS
    if button == glfw.MOUSE_BUTTON_RIGHT:
        mouse_right = action == glfw.PRESS

def cursor_pos_callback(window, x, y):
    global orbit_yaw, orbit_pitch, pan_x, pan_y, last_mouse_x, last_mouse_y
    dx = x - last_mouse_x
    dy = y - last_mouse_y
    last_mouse_x, last_mouse_y = x, y
    if mouse_left:
        orbit_yaw   += dx * 0.4
        orbit_pitch += dy * 0.4
        orbit_pitch  = max(-89, min(89, orbit_pitch))
    if mouse_right:
        pan_x += dx * 0.003 * orbit_dist
        pan_y -= dy * 0.003 * orbit_dist

def scroll_callback(window, xoff, yoff):
    global orbit_dist
    orbit_dist *= (1.0 - yoff * 0.08)
    orbit_dist  = max(0.5, min(50, orbit_dist))

def create_grid(size=5, step=1):
    lines = []
    for i in range(-size, size + 1):
        lines += [i, 0, -size, 0.2, 0.2, 0.3]
        lines += [i, 0,  size, 0.2, 0.2, 0.3]
        lines += [-size, 0, i, 0.2, 0.2, 0.3]
        lines += [ size, 0, i, 0.2, 0.2, 0.3]

    data = np.array(lines, dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))

    glBindVertexArray(0)

    count = (size * 2 + 1) * 4
    return vao, vbo, count

def main():
    glfw.init()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WIDTH, HEIGHT, "hydrogen atom simulator", None, None)
    glfw.make_context_current(window)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    prog = build_program()
    u_mvp = glGetUniformLocation(prog, "uMVP")
    proj = perspective(60.0, WIDTH / HEIGHT, 0.01, 1000.0)

    point = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    vao_point = glGenVertexArrays(1)
    vbo_point = glGenBuffers(1)
    glBindVertexArray(vao_point)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_point)
    glBufferData(GL_ARRAY_BUFFER, point.nbytes, point, GL_STATIC_DRAW)
    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    glBindVertexArray(0)
    glPointSize(10.0)

    vao_grid, vbo_grid, grid_count = create_grid()

    prev = time.time()

    while not glfw.window_should_close(window):
        now = time.time()
        dt = now - prev
        prev = now

        glClearColor(0.03, 0.03, 0.07, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        mv  = look_at_modelview()
        mvp = proj @ mv

        glUseProgram(prog)
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.T.flatten())

        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, grid_count)
        glBindVertexArray(0)

        glBindVertexArray(vao_point)
        glDrawArrays(GL_POINTS, 0, 1)
        glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()