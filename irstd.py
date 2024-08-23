import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import cv2


class IRSTD(Dataset):
    def __init__(self, model, transform, input_image, patch_h, patch_w, overlap_ratio):
        self.model = model
        self.transform = transform
        self.image = input_image

        self.patches_s = []
        step_size_h = int(patch_h * (1 - overlap_ratio))
        step_size_w = int(patch_w * (1 - overlap_ratio))
        for i in range(0, self.image.shape[0], step_size_h):
            for j in range(0, self.image.shape[1], step_size_w):
                end_i = min(i + patch_h, self.image.shape[0])
                end_j = min(j + patch_w, self.image.shape[1])

                patch = self.image[i:end_i, j:end_j]

                if patch.shape[0] < patch_h or patch.shape[1] < patch_w:
                    full_patch = np.zeros((patch_h, patch_w, patch.shape[2]), dtype=patch.dtype)
                    full_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = full_patch

                self.patches_s.append(patch)


    def __len__(self):
        return len(self.patches_s)

    def __getitem__(self, idx):
        patch = self.patches_s[idx]

        patch = self.transform.apply_image(patch)
        patch_torch = torch.as_tensor(patch, device='cuda')
        patch_torch = patch_torch.permute(2, 0, 1).contiguous()[:, :, :]
        patch_torch = self.model.preprocess(patch_torch)

        return patch_torch
