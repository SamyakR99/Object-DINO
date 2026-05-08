"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse
import random
import pickle

import torch
import datetime
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
from object_discovery import ncut 
import matplotlib.pyplot as plt
import time
from torchvision import transforms
# torch.cuda.empty_cache()
from samyak_feature_extraction_copy import extract_multi_layer_features
from transformers import AutoModel
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from dino_seg import dino_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_base",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v", "our_final_layer_heads","samyak"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device('cuda') 
    # model = get_model(args.arch, args.patch_size, device)
    pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name
    )
    model.eval().cuda()

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with TokenCut 
        exp_name = f"TokenCut-{args.arch}"
        if "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running TokenCut on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        
    if args.save_feat_dir is not None : 
        os.mkdir(args.save_feat_dir)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    start_time = time.time() 
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        pil_image = transforms.ToPILImage()(img)
        W, H = pil_image.size
        # resize = transforms.Resize((448, 448))  # Resize to original dimensions (H, W)
        # pil_image_resized = resize(pil_image)
        # processor.do_resize = False
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size
        head_dim = hidden_size // num_heads
        patch_size = model.config.patch_size
        processed_H = inputs['pixel_values'].shape[2]
        processed_W = inputs['pixel_values'].shape[3]
        h_featmap = processed_H // patch_size
        w_featmap = processed_W // patch_size
        
        # breakpoint()
        
        # Debug: print if there's a mismatch
        # if W != processed_W or H != processed_H:
        #     print(f"Image resized by processor: ({W}x{H}) -> ({processed_W}x{processed_H})")
        
        # For visualizations and ncut
        scales = [processed_W, processed_H]
        size_im = (processed_W, processed_H)


        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit"  in args.arch:
                
                def get_qkv_hook(layer_idx):
                    def hook(module, input, output):
                        hidden_states = input[0]
                        q = module.q_proj(hidden_states)
                        k = module.k_proj(hidden_states)
                        v = module.v_proj(hidden_states)

                        qkv_outputs[layer_idx] = {
                            'q': q.detach().cpu(),
                            'k': k.detach().cpu(),
                            'v': v.detach().cpu()
                        }
                    return hook
                
                
                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    qkv_outputs = {}
                    hook_handles = []
                    # Extract the qkv features of the last attention layer
                    for i, layer in enumerate(model.layer):
                        target_layer = layer.attention
                        hook_handle = target_layer.register_forward_hook(get_qkv_hook(i))
                        hook_handles.append(hook_handle)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)

                    for handle in hook_handles:
                        handle.remove()
                    
                    N_patches = (processed_W // model.config.patch_size) * (processed_H // model.config.patch_size)

                    processed_qkv_by_layer = {}
                    for layer_idx, qkv_dict in qkv_outputs.items():
                        q = qkv_dict['q'][:, 5:, :] # CLS + 4Reg
                        k = qkv_dict['k'][:, 5:, :]
                        v = qkv_dict['v'][:, 5:, :]
                        # breakpoint()

                        # Use the dynamically obtained num_heads and head_dim
                        q = q.reshape(1, N_patches, num_heads, head_dim).permute(0, 2, 1, 3)
                        k = k.reshape(1, N_patches, num_heads, head_dim).permute(0, 2, 1, 3)
                        v = v.reshape(1, N_patches, num_heads, head_dim).permute(0, 2, 1, 3)

                        processed_qkv_by_layer[layer_idx] = {
                            'q': q,
                            'k': k,
                            'v': v
                        }

                    last_layer_idx = max(processed_qkv_by_layer.keys())
                    # print(last_layer_idx)
                    # breakpoint()
                    # Modality selection
                    if args.which_features == "k":
                        #feats = k[:, 1:, :]
                        # breakpoint()
                        ko = processed_qkv_by_layer[last_layer_idx]['k']
                        ko_updated = ko.reshape(1,N_patches,-1)
                        feats = ko_updated
                    elif args.which_features == "q":
                        #feats = q[:, 1:, :]
                        qo = [processed_qkv_by_layer[last_layer_idx]['q']]
                        qo_updated = qo.reshape(1,N_patches,-1)
                        feats = qo_updated
                        
                    elif args.which_features == "v":
                        #feats = v[:, 1:, :]
                        vo = [processed_qkv_by_layer[last_layer_idx]['v']]
                        vo_updated = vo.reshape(1,N_patches,-1)
                        feats = vo_updated
                    
                    elif args.which_features == "our_final_layer_heads":
                        ko = processed_qkv_by_layer[last_layer_idx]['k']
                        selcted_heads = [0,1,2,3,6,8,9,10] ## for vitb16
                        ko_selected_heads  = ko[:,selcted_heads,:,:]
                        concatenated_heads = ko_selected_heads
                        feats = concatenated_heads.reshape(1,N_patches,-1)
                    
                    elif args.which_features == "samyak":
                        # ko = processed_qkv_by_layer[last_layer_idx]['k']
                        # qo = processed_qkv_by_layer[last_layer_idx]['q']
                        # vo = processed_qkv_by_layer[last_layer_idx]['v']
                        
                        
                        # k1 = processed_qkv_by_layer[last_layer_idx-1]['k']
                        # selcted_heads = [0,1,2,3,6,8,9,10] ## for vitb16
                        # # selcted_heads = [0,2,3,5] ## for vits16 - final layer
                        # # selcted_heads_prefinal = [0,2,4,5] ## for vits16 - penultimate layer
                        
                        # ko_selected_heads  = ko[:,selcted_heads,:,:]
                        # qo_selected_heads  = qo[:,selcted_heads,:,:]
                        # vo_selected_heads  = vo[:,selcted_heads,:,:]
                        
                        # # k1_selected_heads  = k1[:,selcted_heads_prefinal,:,:]
                        # # concatenated_heads = torch.cat((ko_selected_heads, k1_selected_heads), dim=1)
                        # # concatenated_heads = torch.cat((ko_selected_heads, k1_selected_heads), dim=1)
                        # concatenated_heads = ko_selected_heads
                        # feats = concatenated_heads.reshape(1,N_patches,-1)
                        # breakpoint()
                        
                        ################## Use below for our complete method ##################
                        result = extract_multi_layer_features(processed_qkv_by_layer, num_heads, head_dim)
                        feats = result['features']
                        # print(result['cluster_info'])
                        
                        # sam_k = extract_multi_layer_features(model, img)['features']
                        # breakpoint()
                        # feats = [sam_k]
                        
                    if args.save_feat_dir is not None : 
                        np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
                        continue

            else:
                raise ValueError("Unknown model.")

        # ------------ Apply TokenCut ------------------------------------------- 
        if not args.dinoseg:
            pred, objects, foreground, seed , bins, eigenvector= ncut(feats, [w_featmap, h_featmap], scales, init_image_size, args.tau, args.eps, im_name=im_name, no_binary_graph=args.no_binary_graph)
            
            if args.visualize == "pred" and args.no_evaluation :
                image = dataset.load_image(im_name, size_im)
                visualize_predictions(image, pred, vis_folder, im_name)
            if args.visualize == "attn" and args.no_evaluation:
                visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
            if args.visualize == "all" and args.no_evaluation:
                image = dataset.load_image(im_name, size_im)
                visualize_predictions(image, pred, vis_folder, im_name)
                visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
                        
        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
        
        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        image = dataset.load_image(im_name)
        #visualize_predictions(image, pred, vis_folder, im_name)
        #visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    end_time = time.time()
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)
