from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch

def tensor_to_numpy(tensor):
    """Converts a tensor to numpy array and removes batch/channel dimensions"""
    return tensor.squeeze().cpu().numpy()


def calculate_ssim(target, pred):
    """
    计算结构相似度 (SSIM)
    :param target: ground truth image (numpy)
    :param pred: generated image (numpy)
    :return: average SSIM score across slices
    """
    target = (target - target.min()) / (target.max() - target.min())
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    ssim_score = 0
    depth = target.shape[0]
    for i in range(depth):
        ssim_score += ssim(target[i], pred[i], data_range=1.0)
    return ssim_score / depth


def calculate_psnr(target, pred):
    """
    计算峰值信噪比 (PSNR)
    :param target: ground truth image (numpy)
    :param pred: generated image (numpy)
    :return: PSNR score
    """
    target = (target - target.min()) / (target.max() - target.min())
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    return psnr(target, pred, data_range=1.0)


def evaluate_model(model, dataloader, device, diffusion=None, num_timesteps=1000):
    """
    对整个数据集进行评估，返回平均 SSIM 和 PSNR
    """
    model.eval()
    if diffusion is None:
        from pytorch3dunet.unet3d.diffusion import DiffusionModel
        diffusion = DiffusionModel(num_timesteps=num_timesteps).to(device)

    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = data['image'].to(device)
            targets = data['label'].to(device)

            # 使用扩散过程生成样本
            x = torch.randn_like(inputs)
            pred = diffusion.p_sample_loop(model, x.shape)

            for i in range(pred.shape[0]):
                pred_np = tensor_to_numpy(pred[i])
                target_np = tensor_to_numpy(targets[i])

                total_ssim += calculate_ssim(target_np, pred_np)
                total_psnr += calculate_psnr(target_np, pred_np)
                count += 1

    avg_ssim = total_ssim / count if count else 0
    avg_psnr = total_psnr / count if count else 0

    print(f"[Evaluation] Avg SSIM: {avg_ssim:.4f}, Avg PSNR: {avg_psnr:.4f}")

    return {
        "ssim": avg_ssim,
        "psnr": avg_psnr
    }
