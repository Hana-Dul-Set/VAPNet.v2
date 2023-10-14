import math

import numpy as np

def perturb(input, operator):
    cx, cy, w, h = input
    ox, oy, oz, oa = operator

    cx_out = cx + w * ox
    cy_out = cy + h * oy
    w_out = w + w * oz
    h_out = h + h * oz

    output = [cx_out, cy_out, w_out, h_out]
    if oa != 0:
        return rotate(output, oa)
    else:
        return output_to_bounding_box(output)

def bounding_box_to_input(box):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    input = [cx, cy, w, h]
    return input

def output_to_bounding_box(output):
    cx, cy, w, h = output

    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    
    box = [x1, y1, x2, y2]
    return box

def shifting(box, operator):
    input = bounding_box_to_input(box)
    return perturb(input, operator)

def zooming(box, operator):
    input = bounding_box_to_input(box)
    return perturb(input, operator)

def cropping(box, operator):
    input = bounding_box_to_input(box)
    return perturb(input, operator)

def rotation(box, oa):
    input = bounding_box_to_input(box)
    operator = [0, 0, 0, oa]
    return perturb(input, operator), oa

def rotate(input, oa):
    cx, cy, w, h = input
    cos_a = math.cos(oa)
    sin_a = math.sin(oa)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    c = np.array([cx, cy])
    v_list = np.array([[-w // 2, -h // 2], [w // 2, -h // 2], [w // 2, h // 2], [-w // 2, h // 2]])
    box = []
    for v in v_list:
        r = np.dot(rotation_matrix, v)
        u = (c + r)
        u = u.tolist()
        u[0] = int(u[0])
        u[1] = int(u[1])
        box.append(u)
    return box
