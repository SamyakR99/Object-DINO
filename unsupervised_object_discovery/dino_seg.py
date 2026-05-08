import numpy as np
import torch
import cv2

def dino_seg(attentions, featmap_size, patch_size, head=4):
    """
    DINO segmentation baseline from LOST.

    attentions: list of attention tensors per layer
                each: (heads, tokens, tokens) OR (1, heads, tokens, tokens)
    featmap_size: (w_featmap, h_featmap)
    patch_size: ViT patch size
    head: which attention head to use
    """

    # last layer attention
    attn = attentions[-1]

    if attn.dim() == 4:
        attn = attn[0]  # remove batch

    # select head
    attn = attn[head]

    # CLS → patches attention
    cls_attn = attn[0, 5:]  # drop CLS token
    cls_attn = cls_attn.cpu().numpy()

    w_featmap, h_featmap = featmap_size
    cls_attn = cls_attn.reshape(h_featmap, w_featmap)

    # normalize
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() + 1e-6)

    # threshold
    mask = cls_attn > cls_attn.mean()

    # largest connected component
    mask = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels <= 1:
        return np.zeros((1, 4))  # no object found

    largest = 1 + np.argmax([
        np.sum(labels == i) for i in range(1, num_labels)
    ])

    comp = (labels == largest)

    ys, xs = np.where(comp)

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    # convert to image coords
    x1 *= patch_size
    y1 *= patch_size
    x2 *= patch_size
    y2 *= patch_size

    return np.array([[x1, y1, x2, y2]])
