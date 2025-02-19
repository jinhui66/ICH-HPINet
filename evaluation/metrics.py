import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch
import torch.nn.functional as F
from hausdorff import hausdorff_distance

def dice(annotation, segmentation, void_pixels=None, thresh=0.5):

    assert(annotation.shape == segmentation.shape)
    annotation = annotation.detach().numpy()
    segmentation = segmentation.detach().numpy()

    # annotation = annotation.astype(np.bool)
    # segmentation = segmentation.astype(np.bool)

    # return np.sum(annotation & segmentation) / (np.sum(annotation, dtype=float) + np.sum(segmentation, dtype=float)) * 2

    annotation = annotation>=thresh
    segmentation = segmentation>=thresh

    return 2 * np.sum(annotation & segmentation) / (np.sum(annotation) + np.sum(segmentation))


def jaccard(gt, pred):
    """
    计算Jaccard指数(IoU)
    Args:
        pred: 预测图像
        gt: 真实标签图像
    Returns:
        float: Jaccard指数
    """
    # 转换为float32类型并归一化
    pred = pred.detach().numpy()
    gt = gt.detach().numpy()

    # 二值化
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
    # 二值化
    pred_points = np.array(np.where(pred > 0.5)).T
    gt_points = np.array(np.where(gt > 0.5)).T
    
    # 处理边界情况
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    elif len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf
    
    return hausdorff_distance(pred[0], gt[0])
    
