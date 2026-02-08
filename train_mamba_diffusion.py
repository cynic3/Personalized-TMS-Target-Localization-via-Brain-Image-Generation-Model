import os
import torch
import yaml
import argparse
from pytorch3dunet.unet3d.model import MambaDiffusionUNet
from pytorch3dunet.unet3d.diffusion import DiffusionModel
from dataloader.loader import form_dataloader
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba+Diffusion 3D Image Generation")
    parser.add_argument("--config", type=str, default="config/main_mamba_diffusion.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="weights/mamba_diffusion",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device (cpu or cuda)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # 创建数据集
    train_loader = form_dataloader(config['train_path'], config['img_sz'], config['batch_size'])

    # 初始化模型
    model = MambaDiffusionUNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        feature_dim=config['feature_dim'],
        f_maps=config['f_maps']
    ).to(device)

    # 初始化扩散模型
    diffusion = DiffusionModel(
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        num_timesteps=config['num_timesteps']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = diffusion.p_loss  # 使用扩散损失函数

    # 开始训练
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            images = data['image'].to(device)
            timesteps = torch.randint(0, config['num_timesteps'], (images.shape[0],), device=device).long()

            loss = criterion(model, images, timesteps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}] [Batch {batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

        # 保存模型
        if (epoch + 1) % config['save_inter'] == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch}.pt"))
            print(f"[INFO] Saved checkpoint at epoch {epoch}")

if __name__ == "__main__":
    main()
