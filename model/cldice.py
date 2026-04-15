import numpy as np
from skimage.morphology import skeletonize, remove_small_objects


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    y_true, y_pred: numpy array, binary {0,1}
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def cldice_hard(y_true, y_pred, smooth=1e-6):
    """
    Hard clDice using binary masks and hard skeletonization.
    y_true, y_pred: 2D binary numpy arrays {0,1}
    """
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    skel_pred = skeletonize(y_pred).astype(np.uint8)
    skel_true = skeletonize(y_true).astype(np.uint8)

    sp = skel_pred.sum()
    sg = skel_true.sum()

    # 两个都没有前景骨架：视为完美匹配
    if sp == 0 and sg == 0:
        return 1.0
    # 只有一边有：视为最差
    if sp == 0 or sg == 0:
        return 0.0

    # 拓扑 precision / recall
    tprec = (np.logical_and(skel_pred, y_true).sum() + smooth) / (sp + smooth)
    trec  = (np.logical_and(skel_true, y_pred).sum() + smooth) / (sg + smooth)

    return (2.0 * tprec * trec) / (tprec + trec + smooth)

