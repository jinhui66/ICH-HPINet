import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch
import torch.nn.functional as F
from hausdorff import hausdorff_distance

def dice(annotation, segmentation, void_pixels=None, thresh=0.5):

    assert(annotation.shape == segmentation.shape)
    annotation = annotation.detach().numpy()
    segmentation = segmentation.detach().numpy()

    annotation = annotation>=thresh
    segmentation = segmentation>=thresh

    return 2 * np.sum(annotation & segmentation) / (np.sum(annotation) + np.sum(segmentation))


def jaccard(gt, pred):
    pred = pred.detach().numpy()
    gt = gt.detach().numpy()

    pred = pred > 0.5
    gt = gt > 0.5
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else 0

def MAE(gt, pred):
    mae = F.l1_loss(pred, gt)
    return mae
           
def HD(gt, pred):

    gt = gt.detach().numpy()
    pred = pred.detach().numpy()

    pred_points = np.array(np.where(pred > 0.5)).T
    gt_points = np.array(np.where(gt > 0.5)).T
    
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    elif len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf
    
    return hausdorff_distance(pred, gt)
    
