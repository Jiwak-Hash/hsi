# RGB to Hyperspectral CNN Model

A PyTorch implementation for converting RGB images to hyperspectral images (400-700nm, 30 bands) using Convolutional Neural Networks, trained on the CAVE dataset.

## Project Structure

```
/home/jiwak/Ankit_Sir_Project/
├── cnn_model.py          # CNN model architectures
├── dataset.py            # CAVE dataset loading
├── train.py              # Training script
├── inference.py          # Inference script
├── config.py             # Configuration
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
├── README.md             # This file
├── data/                 # Dataset directory
├── models/               # Model architectures
├── checkpoints/          # Saved checkpoints
└── results/              # Results and outputs
```

## Setup

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /home/jiwak/Ankit_Sir_Project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your CAVE dataset in the following location:
```
/home/jiwak/Downloads/Ankit_Sir_Project /archive
```

The dataset should contain hyperspectral images in one of these formats:
- `.mat` files with hyperspectral data
- Directories with individual band images (PNG/JPG format)

## Usage

### Training

**Using your CAVE dataset:**
```bash
python train.py --epochs 100 --batch-size 4 --lr 0.001
```

**Using synthetic dataset (for testing):**
```bash
python train.py --use-synthetic --epochs 50 --batch-size 4
```

**Advanced options:**
```bash
python train.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 0.001 \
    --architecture HSCNN  # Options: HSCNN, ResidualHSCNN, DenseHSCNN
```

### Inference

```bash
python inference.py \
    --model-path checkpoints/checkpoint_epoch_50.pth \
    --input-image path/to/rgb_image.jpg \
    --output-dir results/ \
    --visualize
```

## Model Architectures

### 1. **HSCNN** (Hyperspectral CNN)
Simple CNN with encoder-decoder structure:
- 5 convolutional layers
- Batch normalization
- ReLU activations
- Output: 30 spectral bands

### 2. **ResidualHSCNN**
CNN with residual connections:
- Improved gradient flow
- Better feature learning
- Faster convergence

### 3. **DenseHSCNN**
Dense connections with feature reuse:
- Dense feature connections
- Better parameter efficiency
- Superior feature propagation

## Specifications

| Parameter | Value |
|-----------|-------|
| Wavelength Range | 400-700 nm |
| Number of Bands | 30 |
| Input Channels | 3 (RGB) |
| Output Channels | 30 (Hyperspectral) |
| Image Size | 256 × 256 |
| Batch Size | 4 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE |

## Training Metrics

The model tracks the following metrics during training:

- **Loss (MSE)**: Mean Squared Error between predicted and ground truth
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **SAM**: Spectral Angle Mapper (degrees)

## Configuration

Edit `config.py` to customize:

```python
# Model
ARCHITECTURE = "HSCNN"  # Model type
NUM_BANDS = 30          # Number of spectral bands
USE_BATCH_NORM = True   # Batch normalization
USE_RESIDUAL = True     # Residual connections

# Training
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 100
LOSS_FUNCTION = "MSE"   # MSE or L1

# Dataset
DATA_PATH = "/path/to/cave/dataset"
TRAIN_TEST_SPLIT = 0.8
VAL_TEST_SPLIT = 0.1
```

## Dataset Support

### CAVE Dataset
- 32 objects with multispectral captures
- 31 bands (400-700nm range)
- Various lighting conditions
- Supported formats: `.mat` files or individual PNG bands

### Custom Datasets
Implement `CAVEDataset` class in `dataset.py`:

```python
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        # Return (rgb_image, hsi_image) as torch tensors
        # rgb_image: (3, H, W)
        # hsi_image: (30, H, W)
        pass
```

## Output

### Training Output
- `checkpoints/`: Model checkpoints saved every 5 epochs
- `results/training_history.png`: Training loss curve
- Best model automatically saved based on validation loss

### Inference Output
- `.npy` file: Raw hyperspectral predictions
- `.png`: Visualization of RGB and selected bands

## Results

Expected performance metrics on CAVE dataset:
- **PSNR**: 25-35 dB
- **SSIM**: 0.85-0.95
- **SAM**: 5-15 degrees

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- numpy, scipy, scikit-image, opencv-python, matplotlib

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch-size 2`
- Reduce image size in `config.py`
- Use CPU: Set `DEVICE = "cpu"` in config.py

### Dataset Loading Issues
- Verify dataset path in `config.py`
- For `.mat` files, check scipy is installed: `pip install scipy`
- Check file format matches expected structure

### Slow Training
- Ensure CUDA is available: Check `nvidia-smi`
- Use `--batch-size` appropriate for GPU memory
- Reduce `image_size` in config.py

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rgb_to_hsi_cnn,
  title={CNN-based RGB to Hyperspectral Image Conversion},
  year={2024}
}
```

## References

- CAVE Dataset: https://www1.cs.columbia.edu/CAVE/databases/multispectral/
- Hyperspectral Imaging: https://en.wikipedia.org/wiki/Hyperspectral_imaging
- CNN Architecture Details: Research papers on spectral recovery

## License

MIT License

## Author

Created for Ankit Sir's Project

## Contact & Support

For issues or questions, check:
1. `README.md` (this file)
2. Inline code comments
3. Configuration in `config.py`
