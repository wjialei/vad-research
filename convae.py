import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLOE
import cv2
from torchvision import transforms
import numpy as np


to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def tensor_to_cv2(tensor_img):
    # 去除 batch 维度（如果有的话）
    if tensor_img.ndim == 4:
        tensor_img = tensor_img.squeeze(0)
    # 从 [C, H, W] → [H, W, C]
    np_img = tensor_img.permute(1, 2, 0).cpu().numpy()
    # 从 [0,1] 转为 [0,255] 并变为 uint8
    np_img = (np_img * 255).astype(np.uint8)
    # RGB → BGR (因为 OpenCV 是 BGR)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return bgr_img


def compute_reconstruction_error(original, reconstructed):
    """
    计算重建误差，这里使用均方误差（MSE）
    """
    return torch.nn.functional.mse_loss(reconstructed, original, reduction='none').cpu().numpy()


def plot_heatmap(image, error_map):
    """
    将误差热力图叠加到原始图像上
    """
    heatmap = np.uint8(255 * error_map / np.max(error_map))  # 归一化到 [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用 Jet 色图
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    return overlay


class ConvAE(nn.Module):
    """论文中的2D卷积自编码器实现"""

    def __init__(self, in_size=256):
        super().__init__()

        # Encoder (论文2.2.1节结构)
        self.encoder = nn.Sequential(
            # Conv1: 512 filters, stride 4
            nn.Conv2d(3, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 第一池化层

            # Conv2: 256 filters
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),

            # Conv3: 128 filters
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),

            # 最终输出128通道的13x13特征图
        )

        # Decoder (逆向结构)
        self.decoder = nn.Sequential(
            # Deconv3
            nn.ConvTranspose2d(128, 256, 3, padding=1),
            nn.ReLU(),

            # Deconv2
            nn.ConvTranspose2d(256, 512, 3, padding=1),
            nn.ReLU(),

            # Unpooling + Deconv1
            nn.Upsample(scale_factor=2),  # 反池化
            nn.ConvTranspose2d(512, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def process_frame_with_convAE(image_bgr, conv_ae_model, device='cuda'):
    """
    完整处理流程：YOLO + 隐私遮挡 + ConvAE + 重建误差
    """
    # Step 1: YOLO 目标检测
    detector_model = YOLOE("yoloe-11l-seg.pt")
    yolo_results = detector_model.predict([image_bgr])

    # Step 2: 隐私遮挡
    masked_bgr = yolo_results[0].plot()

    # Step 3: RGB → Tensor
    image_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = to_tensor(image_rgb).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    # Step 4: 送入 ConvAE
    with torch.no_grad():
        recon_tensor = conv_ae_model(input_tensor)

    # Step 5: 计算重建误差（MSE）
    loss = torch.nn.functional.mse_loss(recon_tensor, input_tensor).item()

    return {
        'masked_image_bgr': masked_bgr,  # OpenCV 图像
        'input_tensor': input_tensor,  # 原始图像 Tensor
        'reconstructed_tensor': recon_tensor,  # 重建图像 Tensor
        'reconstruction_loss': loss  # 异常得分
    }


image_path = "000000574769.jpg"
image_bgr = cv2.imread(image_path)

detector_model = YOLOE("yoloe-11l-seg.pt")
yolo_results = detector_model.predict([image_bgr])

ae_model = ConvAE()
result = process_frame_with_convAE(image_bgr, ae_model, device='cpu')
print(f"anomaly score: {result['reconstruction_loss']}")

cv2.imshow("Masked Image", result['masked_image_bgr'])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result['reconstructed_tensor'].shape)
i_tensor = result['reconstructed_tensor'].squeeze(0)
to_pil_image = transforms.ToPILImage()(i_tensor)
to_pil_image.show()

