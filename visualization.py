# visualization.py

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse


def visualize_slices(volume, output_dir='visualizations', num_slices=8):
    """
    显示 volume 的中间多个切片，并保存为图像
    :param volume: shape = (C, D, H, W)
    :param output_dir: 图像保存路径
    :param num_slices: 显示切片数量
    """
    vol = volume[0]  # 去掉通道维度 (D, H, W)
    depth = vol.shape[0]
    indices = np.linspace(0, depth - 1, num_slices).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for idx, slice_idx in enumerate(indices):
        row = idx // 4
        col = idx % 4
        im = axes[row, col].imshow(vol[slice_idx], cmap='gray')
        axes[row, col].set_title(f'Slice {slice_idx}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization.png'))
    plt.close(fig)
    print(f"[INFO] Visualization saved at {os.path.join(output_dir, 'visualization.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated 3D volumes")
    parser.add_argument("--sample_path", type=str, default="generated_samples/sample_0.npy",
                        help="Path to the generated .npy volume file")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualization images")
    args = parser.parse_args()

    # 加载数据
    sample = np.load(args.sample_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # 可视化
    visualize_slices(sample, output_dir=args.output_dir)
