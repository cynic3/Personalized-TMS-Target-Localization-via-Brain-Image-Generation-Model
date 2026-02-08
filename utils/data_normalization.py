import os
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)
import torch
from os.path import join


 
import os


def diffusion_normal(img):
    """扩散模型专用归一化"""
    img = adaptive_normal(img)  # 原有归一化
    img = (img * 0.5 + 0.5)  # 调整到 [0,1]
    return img


def adaptive_normal(img):
    min_p = 0.001
    max_p = 0.999
    # 1. 处理NaN和负值
    img = torch.where(torch.isnan(img), 0.0, img.float())  # NaN→0
    # 2. 分离前景（非零像素）
    mask_nonzero = img != 0
    foreground = img[mask_nonzero]

    if len(foreground) > 0:
        # 3. 计算分位数边界（支持负值）
        sorted_vals, _ = torch.sort(foreground.flatten())
        n = len(sorted_vals)
        # 下分位数索引
        idx_min = min(max(0, int((n - 1) * min_p + 0.5)), n - 1)
        value_min = sorted_vals[idx_min]
        # 上分位数索引
        idx_max = min(max(0, int((n - 1) * max_p + 0.5)), n - 1)
        value_max = sorted_vals[idx_max]

        # 4. 动态归一化参数
        mean = (value_max + value_min) / 2.0
        stddev = (value_max - value_min) / 2.0 + 1e-8  # 避免除零

        # 5. 仅归一化前景像素
        img[mask_nonzero] = (foreground - mean) / stddev

        # 6. 硬截断并保持原始0值
    img = torch.clamp(img, -1.0, 1.0)
    return img

if __name__ == "__main__":
    data_dir = 'Data_Sample' # 改
    train_sub_dir = sorted(os.listdir(data_dir))
    print("train_sub_dir", train_sub_dir)
    train_files = [{"image": join(data_dir, i, 'input.nii.gz'), 'label': join(data_dir, i, 'output.nii.gz')} for i in train_sub_dir]

    start_transformer = Compose([LoadImaged(keys=['image', 'label']), 
                                EnsureChannelFirstd(keys=['image', 'label']),
                                CropForegroundd(keys=['label'], source_key='label'),
                                ])
    train_file = train_files[0]
    # result = {}
    # result['image'] = nib.load(train_file['image']).get_fdata()
    # result['label'] = nib.load(train_file['label']).get_fdata()
    result = start_transformer(train_file)

    print("Min image: ", result['image'].min())
    print("Min label: ", result['label'].min())

    result['image'] = adaptive_normal(result['image'])
    result['label'] = adaptive_normal(result['label'])

    print("Max image: ", torch.min(result['image']))
    print("Max label: ", torch.min(result['label']))

    print(result['image'].shape)