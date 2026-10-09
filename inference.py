"""Inference script for RGB to Hyperspectral model"""

import torch
import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image as PILImage
    _HAS_CV2 = False
from pathlib import Path
import argparse
from cnn_model import create_model
from utils import get_device, calculate_psnr, calculate_ssim


class Inferencer:
    """Class for inference with trained model"""
    
    def __init__(self, model_path, architecture="HSCNN", num_bands=30, device=None):
        """
        Args:
            model_path: Path to saved model checkpoint
            architecture: Model architecture
            num_bands: Number of spectral bands
            device: Device to use (cuda or cpu)
        """
        
        if device is None:
            device = get_device()
        
        self.device = device
        self.num_bands = num_bands
        
        # Load model
        self.model = create_model(architecture=architecture, num_bands=num_bands).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def infer(self, rgb_image):
        """
        Perform inference on an RGB image
        
        Args:
            rgb_image: Input RGB image (H, W, 3) or path to image file
        
        Returns:
            hsi_image: Hyperspectral image (H, W, num_bands)
        """
        
        # Load image if path is provided
        if isinstance(rgb_image, str):
            if _HAS_CV2:
                rgb_image = cv2.imread(rgb_image)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            else:
                pil = PILImage.open(rgb_image).convert('RGB')
                rgb_image = np.array(pil)
        
        # Normalize
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image.astype(np.float32) / 255.0
        
        # Convert to tensor (1, 3, H, W)
        rgb_tensor = torch.from_numpy(rgb_image).float()
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            hsi_tensor = self.model(rgb_tensor)
        
        # Convert to numpy (H, W, num_bands)
        hsi_image = hsi_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return hsi_image
    
    def infer_batch(self, rgb_images):
        """
        Perform inference on a batch of RGB images
        
        Args:
            rgb_images: Batch of RGB images (B, H, W, 3) or paths
        
        Returns:
            hsi_images: Batch of hyperspectral images (B, H, W, num_bands)
        """
        
        hsi_images = []
        for rgb_image in rgb_images:
            hsi = self.infer(rgb_image)
            hsi_images.append(hsi)
        
        return np.array(hsi_images)


def main():
    parser = argparse.ArgumentParser(description='Inference with RGB to Hyperspectral model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input-image', type=str, required=True, help='Path to input RGB image')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--architecture', type=str, default='HSCNN', help='Model architecture')
    parser.add_argument('--num-bands', type=int, default=31, help='Number of spectral bands')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Inference
    print(f"\nLoading model from {args.model_path}...")
    inferencer = Inferencer(
        model_path=args.model_path,
        architecture=args.architecture,
        num_bands=args.num_bands,
        device=device
    )
    
    print(f"\nPerforming inference on {args.input_image}...")
    hsi_output = inferencer.infer(args.input_image)
    
    print(f"Output shape: {hsi_output.shape}")
    
    # Save output
    output_path = Path(args.output_dir) / 'hsi_output.npy'
    np.save(output_path, hsi_output)
    print(f"Output saved to {output_path}")
    
    # Visualization
    if args.visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Load and show RGB (use cv2 if available, else PIL)
        if _HAS_CV2:
            rgb = cv2.imread(args.input_image)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            pil = PILImage.open(args.input_image).convert('RGB')
            rgb = np.array(pil)

        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Input RGB')
        axes[0, 0].axis('off')

        # Show different bands from HSI output
        band_indices = [0, 10, 20, min(29, hsi_output.shape[2]-1)]
        for idx, band_idx in enumerate(band_indices):
            # compute subplot indices safely
            r = 0 if idx < 3 else 1
            c = idx + 1 if idx < 2 else (idx - 2)
            try:
                ax = axes[r, c]
            except Exception:
                ax = axes.flatten()[idx+1]
            ax.imshow(hsi_output[:, :, band_idx], cmap='viridis')
            ax.set_title(f'Band {band_idx}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / 'visualization.png', dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {Path(args.output_dir) / 'visualization.png'}")
        plt.close()


if __name__ == "__main__":
    main()
