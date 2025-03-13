# Go Games Dataset for PyTorch Neural Network Training

## Overview

This dataset contains Go game positions extracted from high-quality SGF files for training neural networks. The positions are organized into three strength categories based on game quality.

## Dataset Statistics

- **Total SGF Files Processed**: 61149
- **Valid SGF Files**: 0
- **Total Positions**: 29884
- **Processing Time**: 14.90 seconds

## Strength Categories

The dataset is divided into three strength categories:

- **Standard** (Quality 80-85): 2704 games, 9934 positions
- **Strong** (Quality 86-92): 3397 games, 9958 positions
- **Elite** (Quality 93-100): 55048 games, 9992 positions

## Directory Structure

```
dataset/
├── train/
│   ├── boards.pt      # Board state tensors (N, C, H, W)
│   ├── moves.pt       # Move labels (N,)
│   ├── colors.pt      # Player colors (N,)
│   └── metadata.json  # Additional information
├── val/
│   ├── boards.pt
│   ├── moves.pt
│   ├── colors.pt
│   └── metadata.json
├── test/
│   ├── boards.pt
│   ├── moves.pt
│   ├── colors.pt
│   └── metadata.json
├── stats.json         # Processing statistics
└── README.md          # This file
```

## Board Representation

The board state is represented as a tensor with 3 channels:
1. Black stones (1 where black stone is present, 0 elsewhere)
2. White stones (1 where white stone is present, 0 elsewhere)
3. Next player (all 1s if black to play, all 0s if white to play)

## Usage with PyTorch

```python
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader

class GoDataset(Dataset):
    def __init__(self, data_dir):
        self.boards = torch.load(os.path.join(data_dir, "boards.pt"))
        self.moves = torch.load(os.path.join(data_dir, "moves.pt"))
        self.colors = torch.load(os.path.join(data_dir, "colors.pt"))
        
        with open(os.path.join(data_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.moves)
    
    def __getitem__(self, idx):
        return {
            'board': self.boards[idx],
            'move': self.moves[idx],
            'color': self.colors[idx]
        }

# Create datasets
train_dataset = GoDataset('dataset/train')
val_dataset = GoDataset('dataset/val')
test_dataset = GoDataset('dataset/test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)
```

## License

The dataset is intended for research and educational purposes only.

## Creation Date

This dataset was created on 2025.3.13
