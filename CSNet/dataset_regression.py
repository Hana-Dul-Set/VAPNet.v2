import os
import random

import json
import PIL
from PIL import Image
from torch.utils.data import Dataset

from config import Config

Image.MAX_IMAGE_PIXELS = None

# scored crops dataset
class SCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.scored_crops_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'cropped_training_set.json')
            self.random_crops_count = self.cfg.scored_crops_N       
            
        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'cropped_testing_set.json')
            self.random_crops_count = self.cfg.test_crops_N

        self.image_list, self.data_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        score = self.data_list[index]

        return image_name, score

    def build_data_list(self):
        def horizontal_flip_bounding_box(image_size, data):
            bounding_box = data['crop'].copy()
            temp = bounding_box[2]
            bounding_box[2] = image_size[0] - bounding_box[0]
            bounding_box[0] = image_size[0] - temp
            data['crop'] = bounding_box
            return data

        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = [x['name'] for x in data_list]
        score_list = [x['score'] for x in data_list]
        return image_list, score_list

# best crop dataset
class BCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.best_crop_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'best_training_set.json')

        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'best_testing_set.json')

        self.image_list, self.data_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        best_crop_bounding_box = self.data_list[index]
        return image_name, best_crop_bounding_box

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = []
        best_crop_list = []
        for data in data_list:
            image_list.append(data['name'])
            best_crop_list.append(data['crop'])
        return image_list, best_crop_list
    
# unlabeled dataset
class UNDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.unlabeled_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'unlabeled_training_set_20230726_1608.json')

        self.image_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        return image_name

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = []
        for data in data_list:
            image_list.append(data['name'])
        return image_list

if __name__ == '__main__':
    cfg = Config()
    sc_dataset = SCDataset('train', cfg)
    bc_dataset = BCDataset('train', cfg)
    un_dataset = UNDataset('train', cfg)
    print(sc_dataset.__getitem__(0))
    print(bc_dataset.__getitem__(0))
    print(un_dataset.__getitem__(0))