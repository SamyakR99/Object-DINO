"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage

def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    
    """
    feats = feats[0,:,:]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0)) 
    A = A.cpu().numpy()
    
    
    
    # min_value = A.min()
    # if min_value > 0:
    #         tau = min_value * 1.5  # If positive, multiply by 1.5
    # else:
    #     # If negative, avoid making it "worse" by reducing the magnitude or setting a lower bound
    #     tau = min_value * 0.5
    
    # tau = tau.item()
    # breakpoint()
    
    # cls_token = feats[0,0:1,:].cpu().numpy() 
    # features_list = [q, k, v]
    # A_list = []

    # for feat_each in feats:
        
    #     ##old_code
    #     # feat = feat_each[0, 1:, :]
    #     # feat = F.normalize(feat, p=2)
    #     # A_each = feat @ feat.transpose(0, 1) 
        
    #     ## new_code
    #     feat = feat_each[0, :, :]
    #     feat = F.normalize(feat, p=2)
    #     A_each = feat @ feat.transpose(0, 1) 
        
        
        
        
    #     # breakpoint()
    #     # feat = feat_each[0, 1:, :]           # remove CLS token
    #     # # print(f"mean: {torch.mean(feat)}, median: {torch.median(feat)}, std: {torch.std(feat)}, min: {feat.min()}, max: {feat.max()}")
    #     # # Min-max normalization
    #     # # feat = (feat - feat.min()) / (feat.max() - feat.min())
    #     # feat = F.normalize(feat, p=2)  # normalize along dim
    #     # # print(f"mean: {torch.mean(feat)}, median: {torch.median(feat)}, std: {torch.std(feat)}, min: {feat.min()}, max: {feat.max()}")
    #     # # breakpoint()
    #     # # print("feat", feat.shape)
    #     # A_each = feat @ feat.transpose(0, 1)  # compute similarity
    #     # print(A_each.shape)
    #     # print(f"mean: {torch.mean(A_each)}, median: {torch.median(A_each)}, std: {torch.std(A_each)}, min: {A_each.min()}, max: {A_each.max()}")
    #     # # # 
    #     # breakpoint()
    #     # print(torch.mean(A_each), torch.median(A_each), torch.std(A_each), A_each.min(), A_each.max())

    #     A_list.append(A_each)
    # min_value = A_each.min()

    # if min_value > 0:
    #     tau = min_value * 1.5  # If positive, multiply by 1.5
    # else:
    #     # If negative, avoid making it "worse" by reducing the magnitude or setting a lower bound
    #     tau = min_value * 0.5
    
    # tau = tau.item()
    
# Average across q, k, v
    # A = torch.stack(A_list, dim=0).mean(dim=0)
    # A = feats['features']

    # print(tau)
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition 
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition) 
    cc = objects[np.unravel_index(seed, dims)]
    

    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        
        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError


def lost(feats, dims, scales, init_image_size, k_patches=100):
    """
    LOST object discovery from:
    'LOST: Localizing Objects with Self-supervised Transformers'
    """

    w_featmap, h_featmap = dims
    feats = feats.squeeze(0)
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)

    # cosine similarity graph
    A = feats @ feats.t()

    # degree of each patch
    degrees = A.sum(dim=1)

    # select lowest-degree patches
    seed_idx = torch.argsort(degrees)[:k_patches]

    # expand seed
    seed_sim = A[seed_idx].mean(dim=0)

    mask = seed_sim > seed_sim.mean()
    mask = mask.reshape(h_featmap, w_featmap).cpu().numpy()

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.zeros((4,))

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    scale_x = scales[0] / w_featmap
    scale_y = scales[1] / h_featmap

    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    return np.array([x1, y1, x2, y2])
