"""Dataset module for CAVE hyperspectral dataset"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image as PILImage
    _HAS_CV2 = False
from tqdm import tqdm


class CAVEDataset(Dataset):
    """
    CAVE (Colored Objects in a Very Large Workspace) Dataset
    
    The CAVE dataset contains multispectral images of 32 objects captured under
    different lighting conditions, with wavelengths from 400nm to 700nm in 31 bands.
    
    Args:
        data_path: Path to CAVE dataset
        split: 'train', 'val', or 'test'
        transform: Optional image transforms
        image_size: Target image size for resizing
        num_bands: Number of bands to use (default: 30 or 31 depending on data)
    """
    
    def __init__(self, data_path, split='train', transform=None, image_size=256, 
                 num_bands=30, create_rgb=True):
        super(CAVEDataset, self).__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.num_bands = num_bands
        self.create_rgb = create_rgb
        
        self.hyperspectral_images = []
        self.file_names = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load hyperspectral images from CAVE dataset"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
        
        # List all .mat or image files in the dataset
        mat_files = list(self.data_path.glob("*.mat"))
        
        if not mat_files:
            # If no .mat files, look for directory structure
            print("Looking for hyperspectral data in directory structure...")
            self._load_from_directory_structure()
        else:
            self._load_from_mat_files(mat_files)
    
    def _load_from_mat_files(self, mat_files):
        """Load hyperspectral data from .mat files"""
        try:
            import scipy.io as sio
        except ImportError:
            print("scipy required for .mat file loading. Install with: pip install scipy")
            return
        
        for mat_file in tqdm(mat_files, desc="Loading CAVE dataset"):
            try:
                mat_data = sio.loadmat(str(mat_file))
                
                # The CAVE dataset typically stores hyperspectral data in a variable
                # Common variable names: 'hyperspectral', 'img', 'data', etc.
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and len(data.shape) == 3:
                            # Normalize data
                            if data.max() > 1.0:
                                data = data / 255.0
                            
                            self.hyperspectral_images.append(data)
                            self.file_names.append(mat_file.stem)
                            break
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
    
    def _load_from_directory_structure(self):
        """Load hyperspectral data from directory structure with individual band images"""
        
        # CAVE dataset might be organized as directories with individual band images
        # Support nested directory structure (e.g., balloons_ms/balloons_ms/)
        
        def load_from_dir(dir_path, object_name=""):
            """Recursively load band images from a directory"""
            try:
                # Look for PNG/JPG files in this directory
                band_files = sorted(dir_path.glob("*.png")) + sorted(dir_path.glob("*.jpg"))
                
                if band_files:
                    band_images = []
                    for band_file in band_files:
                        # Skip RGB images, only load band images (numbered)
                        if "RGB" in band_file.name or "rgb" in band_file.name:
                            continue
                        try:
                            if _HAS_CV2:
                                img = cv2.imread(str(band_file), cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    # fallback to PIL if cv2 fails to read
                                    pil_img = PILImage.open(str(band_file)).convert('L')
                                    img = np.array(pil_img)
                                img = img.astype(np.float32) / 255.0
                                band_images.append(img)
                            else:
                                # Use PIL fallback when cv2 is not available
                                pil_img = PILImage.open(str(band_file)).convert('L')
                                img = np.array(pil_img).astype(np.float32) / 255.0
                                band_images.append(img)
                        except Exception as e:
                            # Skip files that cannot be read and continue
                            print(f"Warning: skipping unreadable file {band_file}: {e}")
                    
                    # Sort and take first num_bands
                    if len(band_images) >= self.num_bands:
                        band_images = band_images[:self.num_bands]
                        hsi = np.stack(band_images, axis=0)  # Shape: (bands, H, W)
                        hsi = np.transpose(hsi, (1, 2, 0))  # Convert to (H, W, bands)
                        self.hyperspectral_images.append(hsi)
                        name = object_name if object_name else dir_path.name
                        self.file_names.append(name)
                        return True
                
                # If no images found in this directory, try subdirectories
                for subdir in sorted(dir_path.iterdir()):
                    if subdir.is_dir():
                        if load_from_dir(subdir, object_name or subdir.parent.name):
                            return True
                
                return False
            except Exception as e:
                print(f"Error in load_from_dir: {e}")
                return False
        
        # Start loading from data_path
        for item in sorted(self.data_path.iterdir()):
            if item.is_dir():
                load_from_dir(item)
    
    def _create_rgb_from_hsi(self, hsi):
        """Create RGB image from hyperspectral image by selecting 3 bands"""
        
        # Select bands corresponding roughly to R, G, B wavelengths
        # Assuming bands are distributed from 400nm to 700nm
        total_bands = hsi.shape[2]
        
        # Approximate band indices for R, G, B
        # Red: ~650nm, Green: ~550nm, Blue: ~450nm
        red_band = int(total_bands * (650 - 400) / (700 - 400))
        green_band = int(total_bands * (550 - 400) / (700 - 400))
        blue_band = int(total_bands * (450 - 400) / (700 - 400))
        
        rgb = np.stack([
            hsi[:, :, min(red_band, total_bands - 1)],
            hsi[:, :, min(green_band, total_bands - 1)],
            hsi[:, :, min(blue_band, total_bands - 1)]
        ], axis=2)
        
        # Normalize
        rgb = np.clip(rgb, 0, 1)
        
        return rgb
    
    def __len__(self):
        return len(self.hyperspectral_images)
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb: RGB image (3, H, W)
            hsi: Hyperspectral image (30, H, W)
        """
        
        hsi = self.hyperspectral_images[idx]  # Shape: (H, W, num_bands)
        
        # Resize if necessary
        if hsi.shape[0] != self.image_size or hsi.shape[1] != self.image_size:
            # Resize using cv2 if available, otherwise use PIL as fallback
            if _HAS_CV2:
                hsi = cv2.resize(hsi, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            else:
                # hsi is float in [0,1]; for multi-band data (>4 channels) resize each band separately
                h, w, bands = hsi.shape
                if bands <= 4:
                    hsi_uint8 = (hsi * 255.0).astype(np.uint8)
                    pil_img = PILImage.fromarray(hsi_uint8)
                    pil_resized = pil_img.resize((self.image_size, self.image_size), resample=PILImage.BILINEAR)
                    hsi = np.array(pil_resized).astype(np.float32) / 255.0
                else:
                    resized = np.zeros((self.image_size, self.image_size, bands), dtype=np.float32)
                    for bi in range(bands):
                        band = (hsi[:, :, bi] * 255.0).astype(np.uint8)
                        pil_band = PILImage.fromarray(band)
                        pil_resized_band = pil_band.resize((self.image_size, self.image_size), resample=PILImage.BILINEAR)
                        resized[:, :, bi] = np.array(pil_resized_band).astype(np.float32) / 255.0
                    hsi = resized
        
        # Create RGB from HSI
        rgb = self._create_rgb_from_hsi(hsi)
        
        # Apply transforms if provided
        if self.transform:
            rgb = self.transform(rgb)
            hsi = self.transform(hsi)
        
        # Convert to torch tensors and transpose to (C, H, W)
        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
        hsi = torch.from_numpy(np.transpose(hsi, (2, 0, 1))).float()
        
        return rgb, hsi


class SyntheticCAVEDataset(Dataset):
    """
    Synthetic CAVE-like dataset for testing purposes
    Generates synthetic RGB-HSI pairs
    """
    
    def __init__(self, num_samples=100, image_size=256, num_bands=30):
        super(SyntheticCAVEDataset, self).__init__()
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_bands = num_bands
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate synthetic RGB and HSI images"""
        
        # Generate synthetic HSI (smoother transitions across bands)
        hsi = np.zeros((self.image_size, self.image_size, self.num_bands), dtype=np.float32)
        
        for b in range(self.num_bands):
            # Create patterns with wavelength-dependent characteristics
            freq = 2 + (b / self.num_bands) * 4
            hsi[:, :, b] = (
                0.5 + 0.3 * np.sin(2 * np.pi * np.arange(self.image_size)[:, None] * freq / self.image_size) *
                       np.sin(2 * np.pi * np.arange(self.image_size)[None, :] * freq / self.image_size)
            )
        
        # Create RGB by selecting 3 bands
        rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        rgb[:, :, 0] = hsi[:, :, self.num_bands - 1]  # Red-ish
        rgb[:, :, 1] = hsi[:, :, self.num_bands // 2]  # Green-ish
        rgb[:, :, 2] = hsi[:, :, 0]  # Blue-ish
        
        # Normalize
        rgb = np.clip(rgb, 0, 1)
        hsi = np.clip(hsi, 0, 1)
        
        # Convert to torch tensors (C, H, W)
        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
        hsi = torch.from_numpy(np.transpose(hsi, (2, 0, 1))).float()
        
        return rgb, hsi


if __name__ == "__main__":
    # Test with synthetic dataset
    print("Testing synthetic dataset...")
    dataset = SyntheticCAVEDataset(num_samples=10, image_size=256, num_bands=30)
    
    rgb, hsi = dataset[0]
    print(f"RGB shape: {rgb.shape}")
    print(f"HSI shape: {hsi.shape}")
    print(f"Dataset length: {len(dataset)}")
