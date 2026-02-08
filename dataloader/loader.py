import sys; sys.path.append('../')
from scipy.ndimage import zoom
from torch.utils import data
import torch
import os
from os.path import join
from glob import glob
import re
from utils.data_normalization import adaptive_normal
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    CastToTyped,
    Resized,
)
from pathlib import Path

class MRIforming_dataset(data.Dataset):
    def __init__(self, data_path, desired_shape=(160, 160, 96)):
        super(MRIforming_dataset, self).__init__()
        subject_path = os.listdir(data_path)
        self.parrent_path = data_path
        self.sub_path = subject_path
        self.desired_shape = desired_shape

        self.start_transformer = LoadImaged(keys=['image', 'label'],
            reader = "NibabelReader",  # 显式指定读取器
            ensure_channel_first = True  # 可选：直接添加通道维度
            )

        self.transformer = Compose(
            [
                EnsureChannelFirstd(keys=['image', 'label']),
                # CropForegroundd(keys=['label'], source_key='label'),
                Resized(keys=['image', 'label'], spatial_size=desired_shape, mode=['trilinear', 'nearest']),
                # ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=7000, b_min=-1.0, b_max=1.0, clip=True),
                ScaleIntensityRanged(keys=['label'], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0, clip=True),
                CastToTyped(keys=['image', 'label'], dtype=torch.float16),  # 半精度减少内存
                ToTensord(keys=['image', 'label'])
            ])

    def __getitem__(self, index, suffix='.nii.gz'):
        subject = Path(self.parrent_path) / self.sub_path[index]
        input_path = str(subject / f'input{suffix}')
        output_path = str(subject / f'output{suffix}')

        # 打印路径验证（调试用）
        print(f"Loading input: {input_path}")
        print(f"Loading output: {output_path}")


        batch = self.start_transformer(dict(image=input_path, label=output_path))
        print(f"Raw image shape: {batch['image'].shape}")  # 调试：打印原始尺寸
        print(f"Raw label shape: {batch['label'].shape}")

        batch['image'] = batch['image'].squeeze(0)
        batch['label'] = batch['label'].squeeze(0)
        batch = self.transformer(batch)
        print(f"New image shape: {batch['image'].shape}")  # 调试：打印原始尺寸
        print(f"New label shape: {batch['label'].shape}")
        batch['name'] = input_path
        # print(subject)
        return batch

    def __len__(self):
        return len(self.sub_path)


# 核心调用
def form_dataloader(updir, image_size, batch_size, shuffle=True):
    dataset = MRIforming_dataset(updir, image_size)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)
