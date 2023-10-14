import os
import shutil
import math

import cv2
import json
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
import tqdm
import albumentations as A

from CSNet.image_utils.image_preprocess import get_shifted_image, get_zooming_image, get_rotated_image
from CSNet.csnet import get_pretrained_CSNet
from config import Config

Image.MAX_IMAGE_PIXELS = None

device = 'cuda:0'
weight_file = './CSNet/output/weight/0907_10epoch_78_csnet_checkpoint.pth'
csnet = get_pretrained_CSNet(device, weight_file)

def get_csnet_score(image_list, csnet, device):
    image_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tensor = []
    for image in image_list:
        # Grayscale to RGB
        if len(image.getbands()) == 1:
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image, (0, 0, image.width, image.height))
            image = rgb_image
        np_image = np.array(image)
        np_image = cv2.resize(np_image, image_size)
        tensor.append(transformer(np_image))
    tensor = torch.stack(tensor, dim=0)
    tensor = tensor.to(device)
    score_list = csnet(tensor)
    return score_list
    
def make_pseudo_label(image_path):
    
    image = Image.open(image_path).convert('RGB')
    image_name = image_path.split('/')[-1]

    horizontal_shift_magnitude = [x * 0.05 for x in range(-8, 9, 1)]
    vertical_shift_magnitude = [x * 0.05 for x in range(-8, 9, 1)]
    
    pseudo_data_list = []
    magnitude_label_list = []
    perturbed_image_list = []
    
    for h_mag in horizontal_shift_magnitude:
        for v_mag in vertical_shift_magnitude:
            if h_mag ** 2 + v_mag ** 2 > max(horizontal_shift_magnitude) ** 2:
                continue
            pseudo_image = get_shifted_image(image,
                                                [0, 0, image.size[0], image.size[1]],
                                                allow_zero_pixel=True,
                                                option='vapnet',
                                                mag_list=[h_mag, v_mag, 0, 0])

            # get csnet score of each perturbed image
            perturbed_image_list.append(pseudo_image)
            magnitude_label_list.append((round(h_mag, 2), round(v_mag, 2)))

    score_list = get_csnet_score(perturbed_image_list, csnet, device).tolist()
    pseudo_data_list = [(x[0], y, img) for x, y, img in zip(score_list, magnitude_label_list, perturbed_image_list)]
    """
    for i in range(len(score_list)):
        pseudo_data_list.append((score_list[index][0], adjustment_label_list[index], magnitude_label_list[index]))
    """
    # pseudo_data_list.append((score, adjustment_label, magnitude_label))

    # sort in desceding order by csnet score
    pseudo_data_list.sort(reverse=True)

    original_image_score = get_csnet_score([image], csnet, device).item()
    best_adjustment_label = pseudo_data_list[0]
    best_adjustment_score = best_adjustment_label[0]

    """
    print("pseudo_data_list:", pseudo_data_list)
    print("length:", len(pseudo_data_list))
    print("original_image_score:", original_image_score)
    """

    if original_image_score + 0.2 < best_adjustment_score:
        return {
            'name': image_name,
            'magnitude': best_adjustment_label[1]
        }
    else:
        return {
            'name': image_name,
            'magnitude': (0.0, 0.0)
        }
    
def make_annotations_for_unlabeled(image_list, image_dir_path):
    pertubed_cnt = 0
    no_perturbed_cnt = 0
    quadrant_cnt = [0, 0, 0, 0]
    
    for image_name in tqdm.tqdm(image_list):
        image_path = os.path.join(image_dir_path, image_name)

        try:
            annotation = make_pseudo_label(image_path)
            if annotation['magnitude'] != (0.0, 0.0):
                pertubed_cnt += 1
                h_mag = annotation['magnitude'][0]
                v_mag = annotation['magnitude'][1]

                if h_mag > 0 and v_mag >= 0:
                    quadrant_cnt[0] += 1
                elif h_mag <= 0 and v_mag > 0:
                    quadrant_cnt[1] += 1
                elif h_mag < 0 and v_mag <= 0:
                    quadrant_cnt[2] += 1
                else:
                    quadrant_cnt[3] += 1
            else:
                no_perturbed_cnt += 1
            # annotation_list.append(annotation)
            with open('./pseudo_data.csv', 'a') as f:
                f.writelines(f'{annotation}\n')
        except Exception as e:
            print(image_name)
            print(e)
    print(f'perturbed_qudrant_cnt:{quadrant_cnt}')
    print(f'no-perturbed_cnt:{no_perturbed_cnt}')
    """
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set.json', 'w') as f:
        json.dump(annotation_list, f, indent=2)
    """
    return

def csv_to_json_for_unlabeld(csv_path, json_path):
    with open(csv_path, 'r') as f:
        csv_list = list(f.readlines())
    json_list = []
    for line in csv_list:
        json_list.append(eval(line))
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    return

def perturbing_for_labeled_data(image, bounding_box, func):
    output = None
    for i in range(0, 100000):
        if func == 0:
            output = get_shifted_image(image, bounding_box, allow_zero_pixel=False, option='vapnet_test', direction=0)
        elif func == 1:
            output = get_shifted_image(image, bounding_box, allow_zero_pixel=False, option='vapnet_test', direction=1)
        elif func == 3:
            output = get_rotated_image(image, bounding_box, allow_zero_pixel=False, option='vapnet_test')
        if output != None:
            break
    if output == None:
        return None
    perturbed_image, operator, new_box = output
    if func != 3:
        new_box = [
            [new_box[0], new_box[1]],
            [new_box[0], new_box[3]],
            [new_box[2], new_box[3]],
            [new_box[2], new_box[1]],
        ]
    adjustment = [0.0] * 4
    magnitude = [0.0] * 4
    if operator[func] > 0:
        if func == 3:
            adjustment_index = (func - 1) * 2
        else:
            adjustment_index = func * 2
    else:
        if func == 3:
            adjustment_index = (func - 1) * 2 + 1
        else:
            adjustment_index = func * 2 + 1

    adjustment[adjustment_index] = 1.0
    magnitude[adjustment_index] = operator[func] if operator[func] >= 0 else -operator[func]

    return perturbed_image, new_box, adjustment, magnitude

def make_annotation_for_labeled(image_path, bounding_box):
    image = Image.open(image_path)
    image_name = image_path.split('/')[-1].split('.')[0]

    best_crop = image.crop(bounding_box)
    best_crop.save(os.path.join('./data/image/image_labeled_vapnet', image_name + f'_-1.jpg'))
    annotation_list = []
    perturbed_image_cnt = [0, 0, 0]
    box_corners = [
        [bounding_box[0], bounding_box[1]],
        [bounding_box[0], bounding_box[3]],
        [bounding_box[2], bounding_box[3]],
        [bounding_box[2], bounding_box[1]],   
    ]
    func_index = [0, 1]
    i = 0
    while i < len(func_index):
        func = func_index[i]
        output = perturbing_for_labeled_data(image, bounding_box, func)
        if output == None:
            i += 1
            continue
        perturbed_image = output[0]
        new_box = output[1]
        adjustment_label = output[2]
        magnitude_label = output[3]
        perturbed_image_name = image_name + f'_{func}_{perturbed_image_cnt[i]}.jpg'
        annotation = {
            'name': perturbed_image_name,
            'bounding_box': box_corners,
            'perturbed_bounding_box': new_box,
            'suggestion': [1.0],
            'adjustment': adjustment_label,
            'magnitude': magnitude_label
        }
        perturbed_image.save(os.path.join('./data/image/image_labeled_vapnet', perturbed_image_name))
        annotation_list.append(annotation)
        if perturbed_image_cnt[i] < 4:
            perturbed_image_cnt[i] += 1
            i -= 1
        i += 1

    annotation_list.append({
        'name': image_name + f'_-1.jpg',
        'bounding_box': box_corners,
        'perturbed_bounding_box': box_corners,
        'suggestion': [0.0],
        'adjustment': [0.0] * 4,
        'magnitude': [0.0] * 4
    })
    return annotation_list

def make_annotations_for_labeled(data_list, image_dir_path):
    annotation_list = []
    for data in tqdm.tqdm(data_list):
        image_name = data['name']
        bounding_box = data['crop']
        image_path = os.path.join(image_dir_path, image_name)
        annotation_list_one_image = make_annotation_for_labeled(image_path, bounding_box)
        annotation_list += annotation_list_one_image
    with open('./data/annotation/labeled_vapnet/labeled_testing_set.json', 'w') as f:
        json.dump(annotation_list, f, indent=2)
    return

def count_images_by_perturbation(annotation_path):
    # count the pseuo label images by adjustment
    with open(annotation_path, 'r') as f:
        data_list = json.load(f)
    
    cnt = [0, 0, 0, 0, 0]
    for data in data_list:
        suggestion = data['suggestion']
        adjustment = data['adjustment']
        if suggestion == [0.0]:
            cnt[4] += 1
        else:
            print(data)
            cnt[adjustment.index(1.0)] += 1
    perturbed_image_sum = sum(cnt) - cnt[4]
    print(perturbed_image_sum)
    print(cnt)
    return

def remove_duplicated_box(annotation_path):
    data_list = []
    new_data_list = []
    with open(annotation_path, 'r') as f:
        data_list = json.load(f)
    for data in data_list:
        new_box = data['perturbed_bounding_box']
        bounding_box = data['bounding_box']
        flag = False
        for new_data in new_data_list:
            if new_data['bounding_box'] == bounding_box and new_data['perturbed_bounding_box'] == new_box:
                flag = True
                break
        if data['name'].split('_')[-1] == '-1.jpg':
            flag = False
        if flag == False:
            new_data_list.append(data)
    with open(annotation_path, 'w') as f:
        json.dump(new_data_list, f, indent=2)
    image_list = os.listdir('./data/image/image_labeled_vapnet')
    name_list = [x['name'] for x in new_data_list]
    for image in image_list:
        if image not in name_list:
            os.remove(os.path.join('./data/image/image_labeled_vapnet', image))

if __name__ == '__main__':
    cfg = Config()
    data_list = []

    # image_list = os.listdir('./data/open_images')
    

    
    # make pseudo dataset
    json_list = []
    with open('../VAPNet/data/annotation/unlabeled_vapnet/unlabeled_training_set.json', 'r') as f:
        json_list = json.load(f)
    image_list = []
    for data in json_list:
        name = data['name']
        if '_' not in name:
            image_list.append(name)
    print(len(image_list))
    make_annotations_for_unlabeled(image_list, image_dir_path='../VAPNet/data/open_images')
    
    # csv_to_json_for_unlabeld('./pseudo_data_1002.csv', './data/annotation/unlabeled_vapnet/unlabeled_training_set_1002.json')

    """
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002.json', 'r') as f:
        json_list = json.load(f)
    new_json_list = []
    for origin_data in tqdm.tqdm(json_list):
        data = origin_data.copy()
        name = data['name']
        suggestion = data['suggestion']
        adjustment = data['adjustment']
        magnitude = data['magnitude']
        if suggestion == [1.0]:
            image = Image.open(os.path.join('./data/open_images', name))
            if adjustment[0] == 1:
                adjustment[0] = 0.0
                adjustment[1] = 1
                magnitude[1] = magnitude[0]
                magnitude[0] = 0
            elif adjustment[1] == 1:
                adjustment[0] = 1
                adjustment[1] = 0.0
                magnitude[0] = magnitude[1]
                magnitude[1] = 0
            h_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            name = name.split('.')[0] + '_h.jpg'
            data['name'] = name
            data['adjustment'] = adjustment
            data['magnitude'] = magnitude
            h_flip.save(os.path.join('./data/open_images', name))
            new_json_list.append(data)
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002.json', 'r') as f:
        json_list = json.load(f)
    json_list = json_list + new_json_list
    print(len(json_list))
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002_h.json', 'w') as f:
        json.dump(json_list, f, indent=2)
    """
    
    """
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002_h.json', 'r') as f:
        json_list = json.load(f)
    transform = A.Compose([
        A.CLAHE(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.5),
        A.ISONoise(p=0.5),
        A.RandomBrightness(p=0.5)
    ])
    new_json_list = []
    for origin_data in tqdm.tqdm(json_list):
        data = origin_data.copy()
        image_name = data['name']
        suggestion = data['suggestion']
        adjustment = data['adjustment']
        magnitude = data['magnitude']
        if suggestion == [0.0]:
            continue
        try:
            image = cv2.imread(os.path.join('./data/open_images', image_name))
            transformed_image = transform(image=image)['image']
            transformed_image_name = image_name.split('.')[0] + '_a.jpg'
            cv2.imwrite(os.path.join('./data/open_images', transformed_image_name), transformed_image)
            data['name'] = transformed_image_name
            new_json_list.append(data)
        except Exception as e:
            print(e)
            print(data)
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002_h.json', 'r') as f:
        json_list = json.load(f)
    json_list = json_list + new_json_list
    print(len(json_list))
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002_h_a.json', 'w') as f:
        json.dump(json_list, f, indent=2)
    
    """
    
    # count
    """
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set_1002_h_a.json', 'r') as f:
         json_list = json.load(f)
    cnt = [0] * 5
    for data in json_list:
        if data['suggestion'] == [0.0]:
            cnt[4] += 1
        else:
            cnt[data['adjustment'].index(1.0)] += 1
    print(cnt)
    print(sum(cnt))
    """

    """
    labeled_annotation_path = './data/annotation/best_crop/best_testing_set_fixed.json'
    with open(labeled_annotation_path, 'r') as f:
        data_list = json.load(f)
    make_annotations_for_labeled(data_list, './data/image')
    remove_duplicated_box('./data/annotation/labeled_vapnet/labeled_testing_set.json')
    count_images_by_perturbation('./data/annotation/labeled_vapnet/labeled_testing_set.json')
    """