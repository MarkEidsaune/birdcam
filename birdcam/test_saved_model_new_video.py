'''
Pytorch Bird Detection Inference with New Images

For cpu inference use:
    
    python3 test_saved_model_new_imgs.py --device cpu
'''

import os
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
from na_birds_dataset import NABirdsDataset
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class SampleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        paths = []
        for file in os.listdir(img_dir):
            paths.append(os.path.join(img_dir, file))
        self.img_list = paths
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        self.img_path = self.img_list[index]
        self.img = Image.open(self.img_path)
        
        transform = T.ComposeInputOnly([
            T.ToTensorInputOnly()
        ])
        
        img = transform(self.img)
        
        return img

def get_classes(path):
    ds = NABirdsDataset(root=path)
    num_classes = ds.num_classes
    classes = ds.classes
    return classes, num_classes

def get_dataset(path, num_images, seed):
    ds = SampleImageDataset(path)
    torch.manual_seed(seed)
    indices = torch.randint(len(ds), (num_images, ))
    ds_sample = torch.utils.data.Subset(ds, indices)
    return ds_sample

def main(args):
    
    classes, num_classes = get_classes(args.nabirds_data_path)
    
    ds_sample_imgs = get_dataset(args.newimgs_data_path, args.num_images, args.seed)
    
    model = torchvision.models.detection.__dict__[args.model](pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(args.device)
    
    training_checkpoint = torch.load(args.model_path)
    model_checkpoint = training_checkpoint['model']
    
    model.load_state_dict(model_checkpoint)
    
    figure = plt.figure(figsize=(16, 16*args.num_images))
    
    for i, img in enumerate(ds_sample_imgs):
        
        figure.add_subplot(args.num_images, 1, i+1)
        
        plt.imshow(img.permute(1, 2, 0))
        plt.axis('off')
        
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(args.device)])[0]
            
        for i in range(len(prediction['labels'])):
            if prediction['scores'][i] > 0.5:
                pred_label = prediction['labels'][i].cpu()
                pred_label_str = classes[int(pred_label)]
                pred_score = prediction['scores'][i].item()
                pred_score_str = str(round(pred_score, 2))
                pred_box = prediction['boxes'][i].cpu()
                pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box
                pred_width = pred_xmax - pred_xmin
                pred_height = pred_ymax - pred_ymin
                plt.gca().add_patch(
                    Rectangle(
                        (pred_xmin, pred_ymin), 
                        pred_width, 
                        pred_height,
                        fill=False,
                        edgecolor='red', 
                        linewidth=2
                    )
                )
                plt.annotate(
                    pred_label_str + pred_score_str,
                    xy=(pred_xmax, pred_ymin),
                )
    plt.savefig(
        os.path.join(args.output_dir, f'{args.model}_test_new_imgs.png')
    )
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', 
                        type=str, help='model name')
    parser.add_argument('--model-path', default='./checkpoints/fasterrcnn_resnet50_fpn/checkpoint.pth', 
                        type=str, help='model path')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device (use cuda or cpu)')
    parser.add_argument('--nabirds-data-path', default='/media/nvme2/nabirds', 
                        type=str, help='dataset path')
    parser.add_argument('--newimgs-data-path', default='/media/nvme2/birdcam', 
                        type=str, help='dataset path')
    parser.add_argument('--num-images', default=5, type=int, help='number of images to test')
    parser.add_argument('--seed', default=77, type=int, 
                        help='seed for random number generator')
    parser.add_argument('--output-dir', default='./test_inferences', type=str, help='output path')
    args = parser.parse_args()
    main(args)