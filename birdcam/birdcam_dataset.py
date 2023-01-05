import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch

class BirdcamDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.vids = [[f.name, f.path] for f in os.scandir(self.root) if f.is_dir()]
        
        self.imgs = []
        for vid_id, vid_path in self.vids:
            for f in os.scandir(os.path.join(vid_path, 'JPEGImages')):
                if f.is_file():
                    frame = f.name[6:-4]
                    img_id = vid_id + '_' + frame
                    self.imgs.append([img_id, f.path])
        self.imgs = sorted(self.imgs, key=lambda img: img[0])
        self.imgs = [img[1] for img in self.imgs]
        
        self.id2label = {0: 'none', 1: 'bird'}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.n_classes = len(self.id2label)
        
        self.labels, self.bboxes = [], []
        for vid_id, vid_path in self.vids:
            for f in os.scandir(os.path.join(vid_path, 'Annotations')):
                if f.is_file():
                    frame = f.name[6:-4]
                    img_id = vid_id + '_' + frame
                    root = ET.parse(f.path).getroot()
                    children = [child.tag for child in root]
                    if 'object' in children:
                        label = root[5][0].text
                        xmin, ymin, xmax, ymax = [round(float(e.text)) for e in root[5][4][:4]]
                        bbox = [xmin, ymin, xmax, ymax]
                    else:
                        label = 'none'
                        bbox = [0, 0, 1, 1]
                    label_id = self.label2id[label]
                    self.labels.append([img_id, label_id])
                    self.bboxes.append([img_id, bbox])
        self.labels = sorted(self.labels, key=lambda label: label[0])
        self.labels = [label[1] for label in self.labels]
        self.bboxes = sorted(self.bboxes, key=lambda bbox: bbox[0])
        self.bboxes = [bbox[1] for bbox in self.bboxes]
                
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        box = torch.tensor(self.bboxes[idx], dtype=torch.float32)
        boxes = box[None, :]
        
        labels = torch.tensor([self.labels[idx]], dtype=torch.int64)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)