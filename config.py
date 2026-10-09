"""Configuration file for RGB to Hyperspectral CNN model"""

# Dataset and path configurations
DATA_PATH = "/home/projectwork/Ankit_Sir_Project/archive/CAVE Dataset"
CHECKPOINT_DIR = "/home/projectwork/Ankit_Sir_Project/checkpoints"
RESULTS_DIR = "/home/projectwork/Ankit_Sir_Project/results"

# Hyperspectral specifications
NUM_BANDS = 31  # Number of spectral bands (400-700nm, 10nm intervals)
WAVELENGTH_MIN = 400  # nm
WAVELENGTH_MAX = 700  # nm
INPUT_CHANNELS = 3  # RGB
OUTPUT_CHANNELS = 31  # Hyperspectral bands

# Image specifications
IMAGE_SIZE = 256  # Input image size
PATCH_SIZE = 32  # Patch size for patching if needed

# Training configurations
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
TRAIN_TEST_SPLIT = 0.8
VAL_TEST_SPLIT = 0.1

# Model configurations
ARCHITECTURE = "HSCNN"  # Can be: HSCNN, ResNet, DenseNet
USE_BATCH_NORM = True
USE_RESIDUAL = True

# Device
DEVICE = "cuda"  # or "cpu"

# Loss function
LOSS_FUNCTION = "MSE"  # MSE, L1, or SSIM

# Optimization
OPTIMIZER = "Adam"  # Adam, SGD
WEIGHT_DECAY = 0.0001
