import os
from itertools import chain

import cv2
import numpy as np
import PIL
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from config import Config
from csnet import CSNet
from dataset_regression import SCDataset
from image_utils.augmentation import *
from test import test_while_training

def not_convert_to_tesnor(batch):
    return batch

def build_dataloader(cfg):
    sc_dataset = SCDataset('train', cfg)
    sc_loader = DataLoader(dataset=sc_dataset,
                              batch_size=8,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    # return data loader
    return sc_loader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir

        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.sc_loader= build_dataloader(cfg)

        self.sc_batch_size = self.cfg.scored_crops_batch_size
        
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay)

        self.epoch = 0
        self.max_epoch = self.cfg.max_epoch
        
        self.train_iter = 0
        self.sc_iter = 0
        self.sc_loss_sum = 0

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])


    def training(self):
        print('\n======train start======\n')
        for index, data in enumerate(self.sc_loader):
            self.model.train().to(self.device)

            sc_data_list = data
            sc_image_names = [x[0] for x in sc_data_list]

            sc_images = []
            sc_scores = []
            for index, image_name in enumerate(sc_image_names):
                image = Image.open(os.path.join(self.image_dir, image_name))
                sc_images.append(image)
                image_fliped = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                sc_images.append(image_fliped)
                sc_scores.append(data[index][1])
                sc_scores.append(data[index][1])
            
            sc_images, sc_scores = self.shuffle_two_lists_in_same_order(sc_images, sc_scores)
            
            sc_images = self.convert_image_list_to_tensor(list(chain(*sc_images)))
            sc_scores = torch.tensor(list(chain(*sc_scores)))

            sc_loss = self.calculate_mse_loss(sc_images, sc_scores)
            
            loss_log = f'L_SC: {sc_loss.item() if sc_loss != None else 0.0:.5f}'
            print(loss_log)
            
            self.sc_loss_sum += sc_loss.item() if sc_loss != None else 0

            self.optimizer.zero_grad()
            sc_loss.backward()
            self.optimizer.step()

            self.train_iter += 1

            if self.train_iter % 20 == 0:
                ave_sc_loss = self.sc_loss_sum / self.sc_iter
                wandb.log({"sc_loss": ave_sc_loss})
                self.sc_loss_sum = 0
                self.sc_iter = 0
            if self.train_iter % 5000 == 0:
                checkpoint_path = os.path.join(self.cfg.weight_dir, 'regression_checkpoint-weight.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Checkpoint Saved...\n')
                test_while_training()

        print('\n======train end======\n')

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            np_image = np.array(np_image)
            np_image = cv2.resize(np_image, self.cfg.image_size)
            tensor.append(self.transformer(np_image))
        tensor = torch.stack(tensor, dim=0)
        
        return tensor

    def calculate_mse_loss(self, images, scores):
        images = images.to(self.device)
        scores = scores.to(self.device)

        scores = scores.view(scores.shape[0], 1)
        predicted_scores = self.model(images)
        loss = self.loss_fn(predicted_scores, scores)

        return loss

    def run(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.training()

            # save checkpoint
            checkpoint_path = os.path.join(self.cfg.weight_dir, 'regression_checkpoint-weight.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Checkpoint Saved...\n')
            test_while_training()

            epoch_log = 'epoch: %d / %d, lr: %8f' % (self.epoch, self.max_epoch, self.optimizer.param_groups[0]['lr'])
            print(epoch_log)

            self.train_iter = 0
            self.sc_loss_sum = 0
            self.sc_iter = 0

    def shuffle_two_lists_in_same_order(self, list1, list2):
        combined_lists = list(zip(list1, list2))
        random.shuffle(combined_lists)
        shuffled_list1, shuffled_list2 = zip(*combined_lists)
        return list(shuffled_list1), list(shuffled_list2)
            
    def make_image_crops(self, image, crops_list):
        image_crops_list = []
        for crop in crops_list:
            image_crops_list.append(image.crop(crop))
        return image_crops_list

    def augment_pair(self, image_pair, labeled=True):
        pos_image = image_pair[0]
        neg_image = image_pair[1]
        # func_list = [shift_borders, zoom_out_borders, rotation_borders]
        func_list = [shift_borders, zoom_out_borders]
        augment_func = random.choice(func_list)
        if labeled:
            augment_pos_image = augment_func(pos_image)
            augment_neg_image = augment_func(neg_image)
        else:
            augment_pos_image = augment_func(pos_image)
            augment_neg_image = neg_image

        return augment_pos_image, augment_neg_image

if __name__ == '__main__':
    cfg = Config()
    """
    wandb.init(
        # set the wandb project where this run will be logged
        project="three_data_loader",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.learning_rate,
        "architecture": "CNN",
        "dataset": "SC/BC/UN",
        "epochs": cfg.max_epoch,
        "memo": "None"
        }
    )
    """
    model = CSNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.run()

    wandb.finish()

    