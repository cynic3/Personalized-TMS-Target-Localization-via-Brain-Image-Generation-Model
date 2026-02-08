import torch
from pytorch3dunet.unet3d.model import MambaDiffusionUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = MambaDiffusionUNet().to(device)
model.eval()

# 输入尺寸匹配你的数据
x = torch.randn(1, 1, 160, 160, 96).to(device)  # batch_size=1, channel=1, D,H,W
t = torch.randint(0, 1000, (1,)).to(device)

# 测试推理
with torch.no_grad():
    out = model(x, t)
    print("Output shape:", out.shape)
