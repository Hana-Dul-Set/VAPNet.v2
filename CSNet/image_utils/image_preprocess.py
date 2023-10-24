import math
import random

import numpy as np
from PIL import Image

from .perturbation import shifting, zooming, cropping, rotation

def is_not_in_image(boudning_box):
    x1, y1, x2, y2 = boudning_box
    if x1 < 0 or x2 < 0 or x1 >= 1 or x2 >= 1:
        return True
    if y1 < 0 or y2 < 0 or y1 >= 1 or y2 >= 1:
        return True
    return False

def update_operator(type, option='csnet', direction=None):

    # ox, oy, oz, oa
    operator = [0.0, 0.0, 0.0, 0.0]

    if type == 'shift':
        if option == 'csnet':
            operator[0] = random.uniform(-0.4, 0.4)
            operator[1] = random.uniform(-0.4, 0.4)
        if option == 'vapnet' or option=='vapnet_test':
            plus_or_minus = random.randint(0, 1)
            operator[0] = -1 * (-1 if plus_or_minus else 1) * random.uniform(0.05, 0.40)
            plus_or_minus = random.randint(0, 1)
            operator[1] = -1 * (-1 if plus_or_minus else 1) * random.uniform(0.05, 0.40)
            while operator[0] ** 2 + operator[1] ** 2 > 0.4 ** 2:
                operator[0] = -1 * (-1 if plus_or_minus else 1) * random.uniform(0.05, 0.40)
                operator[1] = -1 * (-1 if plus_or_minus else 1) * random.uniform(0.05, 0.40)
            
    elif type == 'zoom':
        if option == 'csnet':
            operator[2] = random.uniform(0, 0.4)
        if option == 'vapnet':
            oz_range = [[-0.048, -0.310], [0.053, 0.818]]
            zoom_in_or_out = random.randint(0, 1)
            operator[2] = oz_range[zoom_in_or_out]

    elif type == 'crop':
        if option == 'csnet':
            operator[2] = random.uniform(math.sqrt(0.5), math.sqrt(0.8))
            operator[0] = random.uniform(-operator[2] / 2, operator[2] / 2)
            operator[1] = random.uniform(-operator[2] / 2, operator[2] / 2)

    elif type == 'rotate':
        if option == 'csnet':
            operator[3] = random.uniform(-math.pi/4, math.pi/4)
        if option == 'vapnet' or option == 'vapnet_test':
            plus_or_minus = random.randint(0, 1)
            operator[3] = -1 * (-1 if plus_or_minus else 1) * random.uniform(math.pi/36, math.pi/4)

    return operator

def get_origin_box(norm_box, image_size):
    new_box = norm_box.copy()
    new_box[0] = norm_box[0] * image_size[0]
    new_box[2] = norm_box[2] * image_size[0]
    new_box[1] = norm_box[1] * image_size[1]
    new_box[3] = norm_box[3] * image_size[1]
    new_box = [int(x) for x in new_box]
    return new_box

def get_shifted_image(image, bounding_box, allow_zero_pixel=False, option='csnet', mag=None, direction=None, mag_list=None):
    norm_box = normalize_box(bounding_box, image.size)
    
    operator = update_operator('shift', option, direction)
    if mag != None:
        operator = [0.0, 0.0, 0.0, 0.0]
        operator[direction] = mag
    
    if mag_list != None:
        operator = mag_list

    new_box = shifting(norm_box, operator)
    
    while allow_zero_pixel == False and is_not_in_image(new_box):
        return None

    new_box = get_origin_box(new_box, image.size)
    new_image = image.crop(new_box)

    if option == 'csnet' or mag != None or mag_list != None:
        return new_image
    if option == 'vapnet':
        return new_image, operator
    if option == 'vapnet_test':
        return new_image, operator, new_box

def get_zooming_image(image, bounding_box, allow_zero_pixel=False, option='csnet', mag=None):
    norm_box = normalize_box(bounding_box, image.size)
    
    operator = update_operator('zoom', option)
    if mag != None:
        operator = [0.0, 0.0, 0.0, 0.0]
        operator[2] = mag

    new_box = zooming(norm_box, operator)

    while allow_zero_pixel == False and is_not_in_image(new_box):
        return None
    
    new_box = get_origin_box(new_box, image.size)
    new_image = image.crop(new_box)

    if option == 'csnet' or mag != None:
        return new_image
    if option == 'vapnet':
        return new_image, operator

def get_cropping_image(image, bounding_box, allow_zero_pixel=False, option='csnet'):
    norm_box = normalize_box(bounding_box, image.size)

    operator = update_operator('crop', option)
    new_box = cropping(norm_box, operator)

    while allow_zero_pixel == False and is_not_in_image(new_box):
        return None

    new_box = get_origin_box(new_box, image.size)
    new_image = image.crop(new_box)

    if option == 'csnet':
        return new_image

def rotate_dot(x, y, oa):
    cos_a = math.cos(oa)
    sin_a = math.sin(oa)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    x, y = np.round(np.dot(rotation_matrix, np.array([x, y])))
    x = int(x)
    y = int(y)
    return x, y

def get_rotated_image(image, bounding_box, allow_zero_pixel=False, option='csnet', input_radian=None):

    def is_not_in_image_rotate(corners, image_size):
        for point in corners:
            if point[0] < 0 or point[1] < 0 or point[0] >= image_size[0] or point[1] >= image_size[1]:
                return True
        return False

    # split cases
    operator = update_operator('rotate', option)
    if input_radian != None:
        operator = [0.0, 0.0, 0.0, 0.0]
        operator[3] = input_radian

    oa = operator[3]
    rotated_box_corners, radian = rotation(bounding_box, oa)
    rotated_box_corners_in_original_image = rotated_box_corners

    # check the rotated image is in original image
    while allow_zero_pixel == False and is_not_in_image_rotate(rotated_box_corners, image.size):
        return None

    # make the rectangle cropped image which contains the rotated image
    rec_corners = (min(x[0] for x in rotated_box_corners), min(x[1] for x in rotated_box_corners), max(x[0] for x in rotated_box_corners), max(x[1] for x in rotated_box_corners))
    norm_rotated_box_corners = [[x[0] - min(y[0] for y in rotated_box_corners), x[1] - min(y[1] for y in rotated_box_corners)] for x in rotated_box_corners]
    rec_image = image.crop(rec_corners)

    # rotate the rectangle image
    rotated_rec_image = rec_image.rotate(radian * (180.0 / math.pi), expand=True)
    rec_centers = [rotated_rec_image.size[0] // 2, rotated_rec_image.size[1] // 2]

    # rotate the rotated image(we want) because the rectangle image was rotated
    rotated_box_corners = [rotate_dot(x[0], x[1], -radian) for x in norm_rotated_box_corners]
    rbc_centers = [(max(x[0] for x in rotated_box_corners) + min(x[0] for x in rotated_box_corners)) // 2, (max(x[1] for x in rotated_box_corners) + min(x[1] for x in rotated_box_corners)) // 2]
    rotated_box_corners = [[x[0] - rbc_centers[0] + rec_centers[0], x[1] - rbc_centers[1] + rec_centers[1]] for x in rotated_box_corners]

    # make bounding box and crop
    bounding_box = [min(x[0] for x in rotated_box_corners), min(x[1] for x in rotated_box_corners), max(x[0] for x in rotated_box_corners), max(x[1] for x in rotated_box_corners)]
    rotated_image = rotated_rec_image.crop(bounding_box)
    if option == 'vapnet_test':
        return rotated_image, operator, rotated_box_corners_in_original_image
    elif option != 'vapnet' or input_radian != None:
        return rotated_image
    else:
        return rotated_image, operator
    
def get_shifted_box(image_size, bounding_box_corners, mag=None):

    operator = [0.0, 0.0, 0.0, 0.0]
    operator[0] = mag[0]
    operator[1] = mag[1]
    min_x = min([x[0] for x in bounding_box_corners])
    max_x = max([x[0] for x in bounding_box_corners])
    min_y = min([x[1] for x in bounding_box_corners])
    max_y = max([x[1] for x in bounding_box_corners])

    diff = image_size[0] * operator[0]
    min_x += diff
    max_x += diff

    diff = image_size[1] * operator[1]
    min_y += diff
    max_y += diff

    new_box = [min_x, min_y, max_x, max_y]
    new_box = [
        (new_box[0], new_box[1]),
        (new_box[0], new_box[3]),
        (new_box[2], new_box[3]),
        (new_box[2], new_box[1]),
    ]

    return new_box

def get_rotated_box(bounding_box, input_radian=None):

    min_x = min(x[0] for x in bounding_box)
    max_x = max(x[0] for x in bounding_box)
    min_y = min(x[1] for x in bounding_box)
    max_y = max(x[1] for x in bounding_box)
    bounding_box = [min_x, min_y, max_x, max_y]
    operator = [0.0, 0.0, 0.0, 0.0]
    operator[3] = input_radian

    oa = operator[3]
    rotated_box_corners, radian = rotation(bounding_box, oa)

    return rotated_box_corners

def normalize_box(box, image_size):
    norm_box = box.copy()
    norm_box[0] = norm_box[0] / image_size[0]
    norm_box[2] = norm_box[2] / image_size[0]
    norm_box[1] = norm_box[1] / image_size[1]
    norm_box[3] = norm_box[3] / image_size[1]
    return norm_box

if __name__ == '__main__':
    image_path = '../../data/sample.jpg'
    bounding_box = [200, 100, 300, 250]
    image = Image.open(image_path)
    image.show()
    print(image.size)

    allow_zero_pixel = False

    box = image.crop(bounding_box)
    box.show()

    rotated_image = get_rotated_image(image, bounding_box, allow_zero_pixel)
    rotated_image.show()

    shifted_image = get_shifted_image(image, bounding_box, allow_zero_pixel)
    shifted_image.show()

    zoomed_out_image = get_zooming_image(image, bounding_box, allow_zero_pixel)
    zoomed_out_image.show()

    cropped_image = get_cropping_image(image, bounding_box, allow_zero_pixel)
    cropped_image.show()