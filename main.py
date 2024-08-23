import numpy as np
import torch
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
from scipy.ndimage import label, center_of_mass
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader

from SAM import sam_model_registry, SamPredictor
from SAM.utils.transforms import ResizeLongestSide
from irstd import IRSTD

import warnings
warnings.filterwarnings('ignore')

def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='dataset/IRDST/real')
    parser.add_argument('--ckpt', type=str, default='weights/mobile_sam.pt')
    parser.add_argument('--seq_idx', type=str, default='1')
    parser.add_argument('--ref_idx', type=str, default='1(1)')
    parser.add_argument('--sam_type', type=str, default='vit_t')
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()

    output_path = os.path.join('outputs', args.seq_idx)
    os.makedirs(output_path, exist_ok=True)

    # Path preparation
    ref_image_path = os.path.join(args.data, 'images', args.seq_idx, args.ref_idx + '.png')
    ref_mask_path = os.path.join(args.data, 'masks', args.seq_idx, args.ref_idx + '.png')
    test_images_path = os.path.join(args.data, 'images', args.seq_idx)

    # Load SAM
    sam_type, sam_ckpt = args.sam_type, args.ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)

    # Load ref images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)


    _, binary_mask = cv2.threshold(ref_mask, 127, 255, cv2.THRESH_BINARY)
    labeled_mask, num_features = label(binary_mask)
    centroid = center_of_mass(binary_mask, labeled_mask, range(1, num_features + 1))
    centroid = centroid[0]

    M, N = ref_image.shape[:2]
    window_h = [ensure_even(M // 6), ensure_even(M // 5), ensure_even(M // 4)] # adjustable parameters
    window_w = [ensure_even(M // 6), ensure_even(M // 5), ensure_even(M // 4)] # adjustable parameters
    target_feats = []
    for h_size, w_size in zip(window_h, window_w):
        center_y, center_x = int(centroid[0]), int(centroid[1])
        start_y = max(center_y - h_size // 2, 0)
        end_y = min(center_y + h_size // 2, ref_image.shape[0])
        start_x = max(center_x - w_size // 2, 0)
        end_x = min(center_x + w_size // 2, ref_image.shape[1])

        cropped_image = ref_image[start_y:end_y, start_x:end_x]
        cropped_mask = binary_mask[start_y:end_y, start_x:end_x]

        mask_s = predictor.set_image(cropped_image, cropped_mask)
        feat_s = predictor.features.squeeze().permute(1, 2, 0)
        mask_s = F.interpolate(mask_s, size=feat_s.shape[0:2], mode="bicubic")
        mask_s = mask_s.squeeze()[0]
        target_feat = feat_s[mask_s > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_feats.append(target_feat)


    # Segment small targets in test images
    for filename in tqdm(os.listdir(test_images_path)):

        test_image_path = os.path.join(test_images_path, filename)
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        overlap_ratio = 0.1
        mask_list = []

        for h_size, w_size, target_feat in zip(window_h, window_w, target_feats):
            patch_size_h, patch_size_w = h_size, w_size
            step_size_h = int(patch_size_h * (1 - overlap_ratio))
            step_size_w = int(patch_size_w * (1 - overlap_ratio))

            input = IRSTD(sam, ResizeLongestSide(sam.image_encoder.img_size), test_image, patch_size_h, patch_size_w,
                          overlap_ratio)
            DatasetLoad = DataLoader(dataset=input, batch_size=5, drop_last=False)

            # Feature Matching
            sim_list = []
            for it, batch in enumerate(DatasetLoad):
                features = sam.image_encoder(batch)
                b, C, h, w = features.shape
                features = features / features.norm(dim=1, keepdim=True)
                features = features.reshape(b, C, h * w)

                sim = target_feat @ features
                sim = sim.reshape(b, 1, h, w)
                sim = F.interpolate(sim, scale_factor=2, mode="bilinear")
                sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=(1024, 1024),
                    original_size=(patch_size_h, patch_size_w)).squeeze()
                if sim.dim() == 2:
                    sim = sim.unsqueeze(0)

                sim_list.append(sim)

            sim_list = torch.cat(sim_list, dim=0)
            tensor_list = [sim_list[i] for i in range(sim_list.size(0))]

            conf_map = torch.zeros(test_image.shape[:2]).cuda()
            count = torch.zeros(test_image.shape[:2]).cuda()

            patch_index = 0
            for i in range(0, test_image.shape[0], step_size_h):
                for j in range(0, test_image.shape[1], step_size_w):
                    end_i = min(i + patch_size_h, test_image.shape[0])
                    end_j = min(j + patch_size_w, test_image.shape[1])

                    conf_map[i:end_i, j:end_j] += tensor_list[patch_index][:end_i - i, :end_j - j]
                    count[i:end_i, j:end_j] += 1
                    patch_index += 1

            count = torch.clamp(count, min=1)
            conf_map = conf_map / count

            # Point Prompt-Centric Focusing
            coordinates = peak_local_max(conf_map.cpu().numpy(), min_distance=20,
                                         threshold_abs=np.max(conf_map.cpu().numpy()) * 0.9)
            for topk_xy_i in coordinates[:, [1, 0]]:
                image_x, image_y = topk_xy_i
                image_start_y = max(image_y - patch_size_h // 2, 0)
                image_end_y = min(image_y + patch_size_h // 2, test_image.shape[0])
                image_start_x = max(image_x - patch_size_w // 2, 0)
                image_end_x = min(image_x + patch_size_w // 2, test_image.shape[1])

                image_patch = test_image[image_start_y:image_end_y, image_start_x:image_end_x]

                local_y = image_y - image_start_y
                local_x = image_x - image_start_x

                local_y = max(0, min(local_y, image_patch.shape[0] - 1))
                local_x = max(0, min(local_x, image_patch.shape[1] - 1))

                predictor.set_image(image_patch)
                mask_patch, scores, logits, _ = predictor.predict(
                    point_coords=np.array([[local_y, local_x]], dtype=np.int64),
                    point_labels=[1],
                    mask_input=None,
                    multimask_output=False
                )
                mask_patch = mask_patch.squeeze()

                mask = np.zeros_like(test_image[:, :, 0])
                mask[image_start_y:image_end_y, image_start_x:image_end_x] = mask_patch

                mask_list.append(mask)

        # Triple-Level Ensemble
        final_mask = np.zeros_like(test_image[:, :, 0])
        total_area_threshold = 0.015 / 100 * (test_image.shape[0] * test_image.shape[1])
        for mask_scale in mask_list:
            if np.sum(mask_scale) < total_area_threshold:
                final_mask += mask_scale

        final_mask[final_mask < 2] = 0
        final_mask[final_mask >= 2] = 255

        mask_output_path = os.path.join(output_path, filename)
        cv2.imwrite(mask_output_path, final_mask)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

def ensure_even(size):
    return size if size % 2 == 0 else size - 1
    

if __name__ == "__main__":
    with torch.no_grad():
        main()
