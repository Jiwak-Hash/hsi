#!/usr/bin/env python3
"""Load a saved HSI cube and export reconstructed RGB + per-band PNGs.

Usage:
  python3 scripts/save_hsi_outputs.py results/hsi_output.npy --out-dir results
"""
import sys
import os
import numpy as np
from PIL import Image


def save_per_band(hsi, out_dir, base_name="hsi"):
    os.makedirs(out_dir, exist_ok=True)
    bands = hsi.shape[2]
    for i in range(bands):
        band = hsi[:, :, i]
        # scale to 0-255
        mn, mx = band.min(), band.max()
        if mx - mn < 1e-8:
            im8 = (band * 0).astype('uint8')
        else:
            im8 = ((band - mn) / (mx - mn) * 255.0).clip(0, 255).astype('uint8')
        Image.fromarray(im8).save(os.path.join(out_dir, f"{base_name}_band_{i:02d}.png"))


def reconstruct_rgb(hsi, wavelengths=None):
    # choose bands for R,G,B (approx 650,550,450 nm)
    bands = hsi.shape[2]
    if wavelengths is None:
        wavelengths = np.linspace(400, 700, bands)

    def idx_for(w):
        return int(np.argmin(np.abs(wavelengths - w)))

    r = hsi[:, :, idx_for(650)]
    g = hsi[:, :, idx_for(550)]
    b = hsi[:, :, idx_for(450)]

    rgb = np.stack([r, g, b], axis=2)
    # normalize to 0..1
    mn, mx = rgb.min(), rgb.max()
    if mx - mn > 1e-8:
        rgb = (rgb - mn) / (mx - mn)
    rgb8 = (rgb * 255.0).clip(0, 255).astype('uint8')
    return rgb8


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/save_hsi_outputs.py <hsi_npy_path> [--out-dir DIR]")
        sys.exit(1)

    hsi_path = sys.argv[1]
    out_dir = 'results'
    if '--out-dir' in sys.argv:
        out_dir = sys.argv[sys.argv.index('--out-dir') + 1]

    if not os.path.isfile(hsi_path):
        print('HSI file not found:', hsi_path)
        sys.exit(2)

    hsi = np.load(hsi_path)
    print('Loaded HSI:', hsi.shape, hsi.dtype)

    bands_dir = os.path.join(out_dir, 'hsi_bands')
    save_per_band(hsi, bands_dir, base_name=os.path.splitext(os.path.basename(hsi_path))[0])
    print('Saved per-band PNGs to', bands_dir)

    # reconstruct RGB
    wavelengths = np.linspace(400, 700, hsi.shape[2])
    rgb = reconstruct_rgb(hsi, wavelengths=wavelengths)
    rgb_file = os.path.join(out_dir, 'hsi_reconstructed_rgb.png')
    Image.fromarray(rgb).save(rgb_file)
    print('Saved reconstructed RGB to', rgb_file)


if __name__ == '__main__':
    main()
