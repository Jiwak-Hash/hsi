"""Training script for RGB to Hyperspectral CNN model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os

import config
from cnn_model import create_model
from dataset import CAVEDataset, SyntheticCAVEDataset
from utils import (calculate_psnr, calculate_ssim, calculate_sam, 
                   save_checkpoint, load_checkpoint, plot_training_history, get_device)


class Trainer:
    """Training class for RGB to Hyperspectral model"""
    
    def __init__(self, model, device, checkpoint_dir, results_dir):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        
        # Loss and optimizer
        if config.LOSS_FUNCTION == "MSE":
            self.criterion = nn.MSELoss()
        elif config.LOSS_FUNCTION == "L1":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        if config.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(model.parameters(), 
                                       lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY)
        else:
            self.optimizer = optim.SGD(model.parameters(), 
                                      lr=config.LEARNING_RATE,
                                      weight_decay=config.WEIGHT_DECAY,
                                      momentum=0.9)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for rgb, hsi_gt in progress_bar:
            rgb = rgb.to(self.device)
            hsi_gt = hsi_gt.to(self.device)
            
            # Forward pass
            hsi_pred = self.model(rgb)
            loss = self.criterion(hsi_pred, hsi_gt)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            psnr = calculate_psnr(hsi_pred, hsi_gt)
            total_psnr += psnr if psnr != float('inf') else 0
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}', 'PSNR': f'{psnr:.2f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        
        return avg_loss, avg_psnr
    
    def validate(self, val_loader):
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for rgb, hsi_gt in progress_bar:
                rgb = rgb.to(self.device)
                hsi_gt = hsi_gt.to(self.device)
                
                hsi_pred = self.model(rgb)
                loss = self.criterion(hsi_pred, hsi_gt)
                
                total_loss += loss.item()
                psnr = calculate_psnr(hsi_pred, hsi_gt)
                total_psnr += psnr if psnr != float('inf') else 0
                
                ssim = calculate_ssim(hsi_pred, hsi_gt)
                total_ssim += ssim
        
        avg_loss = total_loss / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def fit(self, train_loader, val_loader, epochs):
        """Train the model"""
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss, train_psnr = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_psnrs.append(train_psnr)
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            
            print(f"Train Loss: {train_loss:.6f} | Train PSNR: {train_psnr:.2f}")
            print(f"Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, self.checkpoint_dir)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, self.checkpoint_dir)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
            
            # Learning rate scheduling
            self.scheduler.step()
        
        # Plot and save training history (skip if plotting fails)
        history_path = os.path.join(self.results_dir, 'training_history.png')
        try:
            plot_training_history(self.train_losses, self.val_losses, history_path)
        except Exception as e:
            print(f"Could not plot training history: {e}")
        
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train RGB to Hyperspectral CNN model')
    parser.add_argument('--use-synthetic', action='store_true', help='Use synthetic dataset for testing')
    parser.add_argument('--full-dataset', action='store_true', help='Train on full dataset without splitting')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--architecture', type=str, default=config.ARCHITECTURE, 
                       help='Model architecture (HSCNN, ResidualHSCNN, DenseHSCNN)')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Create model
    print(f"\nCreating {args.architecture} model...")
    model = create_model(
        architecture=args.architecture,
        num_bands=config.NUM_BANDS,
        input_channels=config.INPUT_CHANNELS,
        hidden_channels=64,
        use_bn=config.USE_BATCH_NORM
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dataset
    print("\nLoading dataset...")
    if args.use_synthetic:
        print("Using synthetic dataset for testing")
        dataset = SyntheticCAVEDataset(
            num_samples=100,
            image_size=256,
            num_bands=config.NUM_BANDS
        )
    else:
        print(f"Loading CAVE dataset from {config.DATA_PATH}")
        try:
            dataset = CAVEDataset(
                data_path=config.DATA_PATH,
                image_size=256,
                num_bands=config.NUM_BANDS
            )
        except Exception as e:
            print(f"Error loading CAVE dataset: {e}")
            print("Falling back to synthetic dataset...")
            dataset = SyntheticCAVEDataset(
                num_samples=100,
                image_size=256,
                num_bands=config.NUM_BANDS
            )
    
    # Split dataset
    if args.full_dataset:
        # Train on full dataset
        print("Training on FULL dataset (no split)")
        train_dataset = dataset
        # Create a small validation set from the same data for monitoring
        val_dataset = dataset
    else:
        # Standard split
        train_size = int(len(dataset) * config.TRAIN_TEST_SPLIT)
        val_size = int(len(dataset) * config.VAL_TEST_SPLIT)
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    
    print(f"Dataset - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=config.CHECKPOINT_DIR,
        results_dir=config.RESULTS_DIR
    )
    
    # Train
    trainer.fit(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
