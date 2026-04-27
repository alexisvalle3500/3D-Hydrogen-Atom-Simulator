import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
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

def create_grid(size=5, step=1):
    # build a flat grid of lines on the xz plane
    lines = []
    for i in range(-size, size + 1):
        # lines along z axis
        lines += [i, 0, -size, 0.2, 0.2, 0.3]
        lines += [i, 0,  size, 0.2, 0.2, 0.3]
        # lines along x axis
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

    # number of vertices is 4 lines per grid line, 2 vertices each
    count = (size * 2 + 1) * 4
    return vao, vbo, count

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

    # compile shaders
    prog = build_program()
    u_mvp = glGetUniformLocation(prog, "uMVP")

    # build projection matrix
    proj = perspective(60.0, WIDTH / HEIGHT, 0.01, 1000.0)

    # single white point at the origin: x, y, z, r, g, b
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

    # make the point bigger so we can see it
    glPointSize(10.0)

    # create the grid
    vao_grid, vbo_grid, grid_count = create_grid()

    # track time between frames
    prev = time.time()

    while not glfw.window_should_close(window):
        now = time.time()
        dt = now - prev
        prev = now

        # clear to dark blue-black background
        glClearColor(0.03, 0.03, 0.07, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        mv  = look_at_modelview()
        mvp = proj @ mv

        glUseProgram(prog)
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.T.flatten())

        # draw the grid
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, grid_count)
        glBindVertexArray(0)

        # draw the point
        glBindVertexArray(vao_point)
        glDrawArrays(GL_POINTS, 0, 1)
        glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()