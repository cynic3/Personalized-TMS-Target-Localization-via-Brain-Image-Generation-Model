import os
import torch
import numpy as np
from pytorch3dunet.unet3d.model import MambaDiffusionUNet
from pytorch3dunet.unet3d.diffusion import DiffusionModel


def sample_images(model, diffusion, image_size=(160, 160, 96), num_samples=4):
    """
    使用训练好的模型从噪声生成图像
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        fake_data = torch.randn((num_samples, 1, *image_size), device=device)
        for t in reversed(range(diffusion.num_timesteps)):
            t_batch = torch.tensor([t] * num_samples, device=device)
            fake_data = diffusion.p_sample(model, fake_data, t_batch)

        # 反归一化处理
        fake_data = (fake_data + 1) / 2.0  # 转换回 [0,1]
        return fake_data.cpu().numpy()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Generate samples using the trained Mamba-Diffusion model")
    parser.add_argument("--checkpoint", type=str, default="weights/mamba_diffusion/model_epoch_*.pt",
                        help="Path to the model checkpoint file")
    parser.add_argument("--output_dir", type=str, default="generated_samples",
                        help="Directory to save generated samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MambaDiffusionUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    print(f"[INFO] Loaded model from {args.checkpoint}")

    # 初始化扩散过程
    diffusion = DiffusionModel(num_timesteps=1000).to(device)

    print("[INFO] Generating samples...")
    samples = sample_images(model, diffusion)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        np.save(os.path.join(args.output_dir, f"sample_{i}.npy"), sample)

    print(f"[INFO] Generated {len(samples)} samples and saved to {args.output_dir}")
