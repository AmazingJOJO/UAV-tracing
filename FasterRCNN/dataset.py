from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from PIL import Image
import transform as T
from torchvision import transforms
class mydataset(Dataset):
    def __init__(self,root,transform = None):
        self.root = root
        self.transform = transform
        self.imgs = []
        folders = os.listdir(root)
        for folder in folders:
            folder_path = os.path.join(root,folder)
            imgs = os.listdir(folder_path)
            json_path = os.path.join(folder_path,"IR_label.json")
            f = open(json_path,'r',encoding='utf-8')
            m = json.load(f) 
            for i in range(len(imgs) - 2):
                if imgs[i][-3:] == 'jpg':
                    img_path = os.path.join(folder_path,imgs[i])
                    if(m["exist"][i] == 1 and m["gt_rect"][i] != [0,0,0,0]):
                        self.imgs.append([img_path,m["gt_rect"][i]])

        

    def __getitem__(self,idx):
        img_path = self.imgs[idx][0]
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = [1] #only UAV
        xmin = np.float(self.imgs[idx][1][0])
        ymin = np.float(self.imgs[idx][1][1])
        xmax = np.float(self.imgs[idx][1][0] + self.imgs[idx][1][2])
        ymax = np.float(self.imgs[idx][1][1] + self.imgs[idx][1][3])
        boxes.append([xmin,ymin,xmax,ymax])

        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img,target)

        return img,target
    def __len__(self):
        return len(self.imgs)

class testdataset(Dataset):
    def __init__(self,root,transform = None):
        self.root = root
        self.transform = transform
        self.imgs = []
        folders = os.listdir(root)
        for folder in folders:
            folder_path = os.path.join(root,folder)
            imgs = os.listdir(folder_path)
            for i in range(len(imgs) - 2):
                if imgs[i][-3:] == 'jpg':
                    img_path = os.path.join(folder_path,imgs[i])
                    self.imgs.append(img_path)
        

    def __getitem__(self,idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        
        # img = Image.open(img_path).convert("RGB")
        img = T.PILToTensor()(img)[0]
        

        return img
    def __len__(self):
        return len(self.imgs)
    


           