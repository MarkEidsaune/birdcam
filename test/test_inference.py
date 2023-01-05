'''
Pytorch Bird Detection Inference

For cpu inference use:
    
    python3 test_saved_model.py --device cpu
'''

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import birdcam.transforms as T
from birdcam.na_birds_dataset import NABirdsDataset
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def get_transforms():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_dataset(path, num_images, seed):
    ds = NABirdsDataset(root=path, transforms=get_transforms())
    num_classes = ds.num_classes
    classes = ds.classes
    torch.manual_seed(seed)
    indices = torch.randint(len(ds), (num_images,))
    ds_sample = torch.utils.data.Subset(ds, indices)
    return ds_sample, classes, num_classes

def main(args):
    
    ds, classes, num_classes = get_dataset(args.data_path, args.num_images, args.manual_seed)
    
    model = torchvision.models.detection.__dict__[args.model](pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(args.device)
    
    training_checkpoint = torch.load(args.model_path)
    model_checkpoint = training_checkpoint['model']
    
    model.load_state_dict(model_checkpoint)
    
    figure = plt.figure(figsize=(16, 16*args.num_images))
    
    for i, (img, target) in enumerate(ds):
        
        figure.add_subplot(args.num_images, 1, i+1)
        
        label = target['labels'][0]
        label_str = classes[int(label)]
        xmin, ymin, xmax, ymax = target['boxes'][0]
        width = xmax - xmin
        height = ymax - ymin
        
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(args.device)])
        
        pred_labels = prediction[0]['labels'].cpu()
        pred_label = pred_labels[0]
        pred_label_str = classes[int(pred_label)]
        pred_scores = prediction[0]['scores'].cpu()
        pred_score = pred_scores[0]
        pred_boxes = prediction[0]['boxes'].cpu()
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_boxes[0]
        pred_width = pred_xmax - pred_xmin
        pred_height = pred_ymax - pred_ymin
        
        plt.title(
            'Target: {} - Predicted: {} (Score: {})'.format(label_str, pred_label_str, pred_score),
            fontsize=10
        )
        plt.axis('off')
        plt.imshow(img.permute(1, 2, 0))
        plt.gca().add_patch(
            Rectangle(
                (xmin,ymin), 
                width, 
                height,
                fill=False,
                edgecolor='green', 
                linewidth=2
            )
        )
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
    plt.savefig(
        os.path.join(args.output_dir, f'{args.model}_test_imgs.png')
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
    parser.add_argument('--data-path', default='/media/nvme2/nabirds', 
                        type=str, help='dataset path')
    parser.add_argument('--num-images', default=5, type=int, help='number of images to test')
    parser.add_argument('--manual-seed', default=77, type=int, 
                        help='seed for random number generator')
    parser.add_argument('--output-dir', default='./test_inferences', type=str, help='output path')
    args = parser.parse_args()
    main(args)