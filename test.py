import os

from shapely.geometry import Polygon
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score as f1
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from config import Config
from dataset import LabledDataset
from CSNet.image_utils.image_preprocess import get_shifted_box, get_rotated_box
from vapnet import VAPNet

def build_dataloader(cfg):
    labeled_dataset = LabledDataset('test', cfg)
    data_loader = DataLoader(dataset=labeled_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers)
    return data_loader

class Tester(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = os.path.join(self.cfg.image_dir, 'image_labeled_vapnet')

        self.data_loader = build_dataloader(self.cfg)
        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        # self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.batch_size = self.cfg.batch_size

        self.adjustment_count = self.cfg.adjustment_count

        self.suggestion_loss_fn = torch.nn.BCELoss(reduction='mean')
        self.adjustment_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')

        self.data_length = self.data_loader.__len__()

        self.fpr_limit = self.cfg.fpr_limit

        self.suggestion_loss_sum = 0
        self.adjustment_loss_sum = 0
        self.magnitude_loss_sum = 0
        self.auc_score_sum = 0
        self.tpr_score_sum = 0
        self.f1_score_sum = [0] * (self.adjustment_count)
        self.iou_score_sum = 0

    def run(self, custom_threshold=0):
        print('\n======test start======\n')

        total_gt_magnitude_label = np.array([])
        total_predicted_magnitude = np.array([])

        total_gt_perturbed_bounding_box = []
        total_gt_bounding_box = []
        total_image_size = []

        self.model.eval().to(self.device)
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.data_loader), total=self.data_length):
                # data split
                image = data[0].to(self.device)
                image_size = data[1].tolist()
                gt_bounding_box = data[2].tolist()
                gt_perturbed_bounding_box = data[3].tolist()
                gt_magnitude_label = data[6].to(self.device)

                # model inference
                predicted_magnitude = self.model(image.to(self.device))

                # caculate loss
                self.magnitude_loss_sum += self.magnitude_loss_fn(predicted_magnitude, gt_magnitude_label)

                # convert tensor to numpy for using sklearn metrics
                gt_magnitude_label = gt_magnitude_label.to('cpu').numpy()
                predicted_magnitude = predicted_magnitude.to('cpu').numpy()

                total_gt_magnitude_label = self.add_to_total(gt_magnitude_label, total_gt_magnitude_label)
                total_predicted_magnitude = self.add_to_total(predicted_magnitude, total_predicted_magnitude)
                
                total_gt_bounding_box += gt_bounding_box
                total_gt_perturbed_bounding_box += gt_perturbed_bounding_box
                total_image_size += image_size
        
        # get predicted bounding box
        predicted_bounding_box = []
        
        for index, gt_perturbed_box in enumerate(total_gt_perturbed_bounding_box):

            magnitude = total_predicted_magnitude[index]


            predicted_box = get_shifted_box(image_size=total_image_size[index], \
                                            bounding_box_corners=gt_perturbed_box, \
                                            mag=magnitude)
            predicted_bounding_box.append(predicted_box)
            

        # calculate average iou score for each bounding box pairs
        iou_score = self.calculate_ave_iou_score(total_gt_bounding_box, predicted_bounding_box)

        print('\n======test end======\n')

        ave_magnitude_loss = self.magnitude_loss_sum  / self.data_length

        loss_log = f'{ave_magnitude_loss}'
        accuracy_log = f'{iou_score:.5f}'
    
        print(loss_log)
        print(accuracy_log)
        
        wandb.log({"Test/test_magnitude_loss": ave_magnitude_loss})
        wandb.log({
            f"Test/IoU": iou_score
        })
    
    def add_to_total(self, target_np_array, total_np_array):
        if total_np_array.shape == (0,):
            total_np_array = target_np_array
        else:
            total_np_array = np.concatenate((total_np_array, target_np_array))
        return total_np_array


    def calculate_suggestion_accuracy(self, gt_suggestion, predicted_suggestion):
        def find_idx_for_fpr(fpr):
            idices = np.where(np.abs(fpr - self.fpr_limit) == np.min(np.abs(fpr - self.fpr_limit)))
            return np.max(idices)

        gt_suggestion = np.array(gt_suggestion).flatten()
        predicted_suggestion = predicted_suggestion.flatten()
        fpr, tpr, cut = roc_curve(gt_suggestion, predicted_suggestion)
        auc_score = auc(fpr, tpr)
        idx = find_idx_for_fpr(fpr)

        tpr_score = tpr[idx]
        threshold = cut[idx]
        """
        print('gt suggestion:', gt_suggestion)
        print('predicted_suggestion:', predicted_suggestion)
        print('FPR:', fpr)
        print('TPR:', tpr)
        print('CUT:', cut)
        print(f'idx: {idx}/threshold: {threshold}')
        """
        return auc_score, tpr_score, threshold
    
    def convert_array_to_one_hot_encoded(self, array):
        largest_value = np.max(array)
        one_hot_encoded = np.zeros_like(array)
        one_hot_encoded[array == largest_value] = 1
        return one_hot_encoded
    
    def calculate_f1_score(self, gt_adjustment, predicted_adjustment):
        def convert_one_hot_encoded_to_index(array):
            if np.all(array == 0):
                return np.array(self.adjustment_count)
            one_index = np.where(array == 1)[0][0]
            return one_index
        
        if len(gt_adjustment) == 0:
            return [0.0] * self.adjustment_count
        
        # print('gt adjustment:', gt_adjustment)
        # print('predicted adjustment:', predicted_adjustment)
        one_hot_encoded_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, axis=1, arr=predicted_adjustment)
        # print('one_hot_predicted adjustment:', one_hot_encoded_adjustment)
        gt_label_list = np.apply_along_axis(convert_one_hot_encoded_to_index, axis=1, arr=gt_adjustment)
        predicted_label_list = np.apply_along_axis(convert_one_hot_encoded_to_index, axis=1, arr=one_hot_encoded_adjustment)
        # print('gt label:', gt_label_list)
        # print('predicted label:', predicted_label_list)
        labels = [i for i in range(0, self.adjustment_count + 1)]
        f1_score = f1(gt_label_list, predicted_label_list, labels=labels, average=None, zero_division=0.0)

        # remove no-suggestion case
        f1_score = f1_score[:-1]
        # print('f1 score:', f1_score)
        return f1_score
        
    def calculate_ave_iou_score(self, boudning_box_list, perturbed_box_list):
        # box format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (counter-clockwise order)
        def calculate_iou_score(box1, box2):
            # print("gt_box:", box1, "/predicted_box:", box2)
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            if poly1.intersects(poly2) == False:
                return 0
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            # print(intersection_area, union_area)

            iou = intersection_area / union_area if union_area > 0 else 0.0
            return iou
        
        iou_sum = 0
        for i in range(len(boudning_box_list)):
            iou_sum += calculate_iou_score(boudning_box_list[i], perturbed_box_list[i])
        
        ave_iou = iou_sum / len(boudning_box_list)
        return ave_iou

def test_while_training(threshold=0):
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run(custom_threshold=threshold)

if __name__ == '__main__':
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file, map_location='cpu'))

    tester = Tester(model, cfg)
    tester.run()