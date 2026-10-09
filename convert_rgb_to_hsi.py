"""Convert RGB image to 31-band hyperspectral image matching CAVE dataset format"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def create_spectral_response(wavelength, wavelength_min=400, wavelength_max=700):
    """
    Create spectral response functions for RGB channels
    Based on CIE standard observer and Bayer filter characteristics
    """
    # Normalized spectral response for standard RGB sensors
    # Red channel peaks around 650nm
    # Green channel peaks around 550nm  
    # Blue channel peaks around 450nm
    
    red_center = 650
    green_center = 550
    blue_center = 450
    bandwidth = 100  # FWHM
    
    red_response = np.exp(-((wavelength - red_center) ** 2) / (2 * (bandwidth / 4) ** 2))
    green_response = np.exp(-((wavelength - green_center) ** 2) / (2 * (bandwidth / 4) ** 2))
    blue_response = np.exp(-((wavelength - blue_center) ** 2) / (2 * (bandwidth / 4) ** 2))
    
    return red_response, green_response, blue_response


def rgb_to_hsi_spectral_reconstruction(rgb_image, num_bands=31, 
                                        wavelength_min=400, wavelength_max=700):
    """
    Convert RGB image to hyperspectral by spectral reconstruction
    
    Args:
        rgb_image: RGB image (H, W, 3) with values in [0, 255] or [0, 1]
        num_bands: Number of spectral bands to generate (default: 31)
        wavelength_min: Minimum wavelength in nm
        wavelength_max: Maximum wavelength in nm
    
    Returns:
        hsi_image: Hyperspectral image (H, W, num_bands) with values in [0, 1]
    """
    
    # Normalize RGB to [0, 1] if needed
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_image = rgb_image.astype(np.float32)
    
    height, width = rgb_image.shape[:2]
    
    # Generate wavelength array
    wavelengths = np.linspace(wavelength_min, wavelength_max, num_bands)
    
    # Get spectral response functions
    red_response, green_response, blue_response = create_spectral_response(wavelengths)
    
    # Normalize responses
    red_response = red_response / (red_response.sum() + 1e-6)
    green_response = green_response / (green_response.sum() + 1e-6)
    blue_response = blue_response / (blue_response.sum() + 1e-6)
    
    # Extract RGB channels
    r_channel = rgb_image[:, :, 0]
    g_channel = rgb_image[:, :, 1]
    b_channel = rgb_image[:, :, 2]
    
    # Initialize hyperspectral image
    hsi_image = np.zeros((height, width, num_bands), dtype=np.float32)
    
    # For each band, create a mixture based on spectral responses
    for band_idx in range(num_bands):
        # Weight each RGB channel by its spectral response at this wavelength
        band_image = (
            r_channel * red_response[band_idx] +
            g_channel * green_response[band_idx] +
            b_channel * blue_response[band_idx]
        )
        
        # Add spectral details through local variation
        # This creates realistic spectral signatures with smooth transitions
        hsi_image[:, :, band_idx] = band_image
    
    # Add spectral smoothing to create realistic band correlations
    for i in range(1, num_bands - 1):
        hsi_image[:, :, i] = 0.25 * hsi_image[:, :, i-1] + \
                             0.5 * hsi_image[:, :, i] + \
                             0.25 * hsi_image[:, :, i+1]
    
    # Normalize to [0, 1]
    hsi_image = np.clip(hsi_image, 0, 1)
    
    return hsi_image, wavelengths


def rgb_to_hsi_interpolation(rgb_image, num_bands=31, 
                              wavelength_min=400, wavelength_max=700):
    """
    Alternative method: Interpolate RGB across spectral bands
    Creates a smooth 31-band representation from RGB
    """
    
    # Normalize RGB to [0, 1]
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_image = rgb_image.astype(np.float32)
    
    height, width = rgb_image.shape[:2]
    
    # Generate wavelength array
    wavelengths = np.linspace(wavelength_min, wavelength_max, num_bands)
    
    # Create a smooth spectral curve through RGB samples
    # Assume RGB samples at wavelengths: B=450, G=550, R=650
    wavelength_samples = np.array([450, 550, 650])
    
    hsi_image = np.zeros((height, width, num_bands), dtype=np.float32)
    
    # For each pixel, interpolate spectral values
    for h in range(height):
        for w in range(width):
            # Get RGB values at this pixel
            b_val = rgb_image[h, w, 0]
            g_val = rgb_image[h, w, 1]
            r_val = rgb_image[h, w, 2]
            
            # Create interpolation function
            f = interp1d(wavelength_samples, [b_val, g_val, r_val], 
                         kind='cubic', fill_value='extrapolate')
            
            # Interpolate across all bands
            hsi_image[h, w, :] = f(wavelengths)
    
    # Clip to valid range
    hsi_image = np.clip(hsi_image, 0, 1)
    
    return hsi_image, wavelengths


def visualize_hsi(hsi_image, wavelengths, output_path=None):
    """Create visualization of HSI bands"""
    
    # Select bands for visualization (roughly R, G, B)
    num_bands = hsi_image.shape[2]
    
    # Get indices for red, green, blue wavelengths
    red_idx = np.argmin(np.abs(wavelengths - 650))
    green_idx = np.argmin(np.abs(wavelengths - 550))
    blue_idx = np.argmin(np.abs(wavelengths - 450))
    
    # Create RGB from these bands
    rgb_from_hsi = np.stack([
        hsi_image[:, :, red_idx],
        hsi_image[:, :, green_idx],
        hsi_image[:, :, blue_idx]
    ], axis=2)
    
    # Normalize for display
    rgb_from_hsi = np.clip(rgb_from_hsi, 0, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original RGB reconstruction
    axes[0, 0].imshow(rgb_from_hsi)
    axes[0, 0].set_title('HSI RGB Reconstruction')
    axes[0, 0].axis('off')
    
    # Show individual bands
    band_samples = [0, num_bands//4, num_bands//2, 3*num_bands//4, num_bands-1]
    for idx, band_idx in enumerate(band_samples[:4]):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(hsi_image[:, :, band_idx], cmap='gray')
        axes[row, col].set_title(f'Band {band_idx} ({wavelengths[band_idx]:.0f}nm)')
        axes[row, col].axis('off')
    
    # Spectral signature plot
    center_h, center_w = hsi_image.shape[0] // 2, hsi_image.shape[1] // 2
    spectral_sig = hsi_image[center_h, center_w, :]
    axes[1, 2].plot(wavelengths, spectral_sig, linewidth=2)
    axes[1, 2].set_xlabel('Wavelength (nm)')
    axes[1, 2].set_ylabel('Reflectance')
    axes[1, 2].set_title('Spectral Signature (Center Pixel)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Convert RGB image to 31-band HSI')
    parser.add_argument('--input-image', type=str, required=True, 
                        help='Path to input RGB image')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for HSI file')
    parser.add_argument('--method', type=str, default='spectral', 
                        choices=['spectral', 'interpolation'],
                        help='Conversion method')
    parser.add_argument('--num-bands', type=int, default=31,
                        help='Number of spectral bands')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization')
    
    args = parser.parse_args()
    
    # Load RGB image
    print(f"Loading RGB image from {args.input_image}...")
    rgb_image = cv2.imread(args.input_image)
    
    if rgb_image is None:
        print(f"Error: Could not load image from {args.input_image}")
        return
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    print(f"Input image shape: {rgb_image.shape}")
    
    # Resize to match CAVE dataset size (1024x1024) if needed
    if rgb_image.shape != (1024, 1024, 3):
        print(f"Resizing image to 1024x1024...")
        rgb_image = cv2.resize(rgb_image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    # Convert RGB to HSI
    print(f"Converting RGB to {args.num_bands}-band HSI using {args.method} method...")
    
    if args.method == 'spectral':
        hsi_image, wavelengths = rgb_to_hsi_spectral_reconstruction(
            rgb_image, num_bands=args.num_bands
        )
    else:
        hsi_image, wavelengths = rgb_to_hsi_interpolation(
            rgb_image, num_bands=args.num_bands
        )
    
    print(f"Generated HSI shape: {hsi_image.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save HSI as .npy file
    output_file = output_dir / 'hsi_ground_truth.npy'
    np.save(output_file, hsi_image)
    print(f"HSI saved to {output_file}")
    
    # Save individual bands as PNG (CAVE format)
    bands_dir = output_dir / 'hsi_bands'
    bands_dir.mkdir(exist_ok=True)
    
    print("Saving individual bands...")
    for band_idx in range(args.num_bands):
        band_image = (hsi_image[:, :, band_idx] * 255).astype(np.uint8)
        band_file = bands_dir / f'band_{band_idx:02d}.png'
        cv2.imwrite(str(band_file), band_image)
    
    print(f"Bands saved to {bands_dir}")
    
    # Save wavelength information
    wavelength_file = output_dir / 'wavelengths.npy'
    np.save(wavelength_file, wavelengths)
    print(f"Wavelengths saved to {wavelength_file}")
    
    # Create visualization
    if args.visualize:
        viz_file = output_dir / 'hsi_conversion_visualization.png'
        visualize_hsi(hsi_image, wavelengths, str(viz_file))
    
    print("\nConversion complete!")
    print(f"HSI format: {hsi_image.shape} (Height × Width × Bands)")
    print(f"Value range: [{hsi_image.min():.3f}, {hsi_image.max():.3f}]")
    print(f"Wavelength range: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")


if __name__ == '__main__':
    main()
