import os
from PIL import Image
import torch

class NABirdsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.imgs = []
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                img_path = os.path.join(os.path.join(root, f'images/{pieces[1]}'))
                self.imgs.append(img_path)
                
        self.class_hierarchy = {}
        with open(os.path.join(self.root, 'hierarchy.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                class_id = int(pieces[0])
                parent_id = int(pieces[1])
                self.class_hierarchy[class_id] = parent_id
                
        self.classes = {}
        with open(os.path.join(self.root, 'classes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                class_id = int(pieces[0])
                self.classes[class_id] = ' '.join(pieces[1:])
        self.num_classes = len(self.classes)
        
        self.labels = []
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                self.labels.append(int(pieces[1]))
                
        self.bboxes = []
        with open(os.path.join(self.root, 'bounding_boxes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                bbox = list(map(int, pieces[1:]))
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                self.bboxes.append(bbox)
                
        self.sizes = []
        with open(os.path.join(self.root, 'sizes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                width, height = pieces[1:]
                self.sizes.append([width, height])
                
    def get_height_and_width(self, idx):
        return self.sizes[idx][1], self.sizes[idx][0]
                
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
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