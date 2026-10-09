"""Utility functions for training and evaluation"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import os


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: Predicted image (tensor or numpy)
        img2: Ground truth image (tensor or numpy)
    
    Returns:
        psnr: PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: Predicted image (tensor)
        img2: Ground truth image (tensor)
        window_size: Window size for SSIM calculation
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        ssim: SSIM value
    """
    
    # Ensure tensors are on CPU
    if img1.is_cuda:
        img1 = img1.cpu()
    if img2.is_cuda:
        img2 = img2.cpu()
    
    # Create Gaussian kernel
    x = np.arange(-window_size // 2 + 1, window_size // 2 + 1, dtype=np.float32)
    gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    # Create 2D Gaussian kernel by outer product
    kernel_2d = np.outer(gauss, gauss)
    kernel = torch.from_numpy(kernel_2d).unsqueeze(0).unsqueeze(0)
    
    # Calculate SSIM for each channel and average
    ssim_values = []
    
    for i in range(img1.shape[1]):  # Iterate over channels
        img1_ch = img1[:, i:i+1, :, :]
        img2_ch = img2[:, i:i+1, :, :]
        
        # Apply convolution with Gaussian kernel
        mu1 = F.conv2d(img1_ch, kernel, padding=window_size // 2)
        mu2 = F.conv2d(img2_ch, kernel, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1_ch ** 2, kernel, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2_ch ** 2, kernel, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1_ch * img2_ch, kernel, padding=window_size // 2) - mu1_mu2
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        
        ssim_ch = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                  ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        ssim_values.append(ssim_ch.mean().item())
    
    return np.mean(ssim_values)


def calculate_sam(img1, img2, eps=1e-8):
    """
    Calculate Spectral Angle Mapper (SAM)
    
    Args:
        img1: Predicted image (B, C, H, W)
        img2: Ground truth image (B, C, H, W)
        eps: Small epsilon value for numerical stability
    
    Returns:
        sam: Mean SAM value in degrees
    """
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Reshape to (B*H*W, C)
    b, c, h, w = img1.shape
    img1_reshaped = img1.transpose(0, 2, 3, 1).reshape(-1, c)
    img2_reshaped = img2.transpose(0, 2, 3, 1).reshape(-1, c)
    
    # Calculate angles
    dots = (img1_reshaped * img2_reshaped).sum(axis=1)
    norms1 = np.sqrt((img1_reshaped ** 2).sum(axis=1))
    norms2 = np.sqrt((img2_reshaped ** 2).sum(axis=1))
    
    norms_product = norms1 * norms2
    norms_product = np.where(norms_product == 0, eps, norms_product)
    
    cos_angles = dots / norms_product
    cos_angles = np.clip(cos_angles, -1, 1)
    
    angles = np.arccos(cos_angles)
    sam_mean = np.mean(angles) * 180.0 / np.pi
    
    return sam_mean


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from epoch {epoch} with loss {loss:.6f}")
    
    return model, optimizer, epoch, loss


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training history"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — skipping training history plot")
        return

    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")

    plt.show()


def visualize_prediction(rgb, prediction, ground_truth, save_path=None):
    """Visualize RGB input, predicted HSI, and ground truth HSI"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy if needed
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if rgb.ndim == 4:
        rgb = rgb[0]
        prediction = prediction[0]
        ground_truth = ground_truth[0]
    
    # RGB visualization
    if rgb.shape[0] == 3:
        rgb_img = np.transpose(rgb, (1, 2, 0))
        axes[0].imshow(rgb_img)
    else:
        axes[0].imshow(rgb[0], cmap='gray')
    axes[0].set_title('Input RGB')
    axes[0].axis('off')
    
    # Prediction visualization (first band)
    axes[1].imshow(prediction[0], cmap='viridis')
    axes[1].set_title('Predicted HSI (Band 1)')
    axes[1].axis('off')
    
    # Ground truth visualization (first band)
    axes[2].imshow(ground_truth[0], cmap='viridis')
    axes[2].set_title('Ground Truth HSI (Band 1)')
    axes[2].axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()


def get_device():
    """Get available device (CUDA or CPU)"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device
