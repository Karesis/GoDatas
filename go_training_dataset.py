#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go Games Dataset Builder for PyTorch Training

Creates a neural network training dataset from organized SGF files with 3 strength categories.
Features a simple, PyTorch-ready format with multi-processing support.
"""

import os
import sys
import argparse
import logging
import json
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("go_dataset_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("go_dataset_builder")

# Constants
BOARD_SIZE = 19  # Standard Go board size
# Strength categories (all games are 80+ quality)
STRENGTH_CATEGORIES = {
    'standard': (80, 85),  # Quality 80-85
    'strong': (86, 92),    # Quality 86-92
    'elite': (93, 100)     # Quality 93-100
}


class SGFParser:
    """Simple SGF parser for Go games"""
    
    @staticmethod
    def parse_sgf(file_path):
        """
        Parse an SGF file and extract game data
        
        Args:
            file_path (Path): Path to the SGF file
            
        Returns:
            dict: Game data including moves, metadata, and board states
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            import re
            
            # Extract metadata
            metadata = {}
            properties = {
                'event': r'EV\[([^\]]*)\]',
                'date': r'DT\[([^\]]*)\]',
                'black_player': r'PB\[([^\]]*)\]',
                'white_player': r'PW\[([^\]]*)\]',
                'result': r'RE\[([^\]]*)\]',
                'black_rank': r'BR\[([^\]]*)\]',
                'white_rank': r'WR\[([^\]]*)\]',
                'komi': r'KM\[([^\]]*)\]',
                'handicap': r'HA\[([^\]]*)\]'
            }
            
            for key, pattern in properties.items():
                match = re.search(pattern, content)
                if match:
                    metadata[key] = match.group(1).strip()
            
            # Extract moves
            black_moves = re.findall(r';B\[([^\]]*)\]', content)
            white_moves = re.findall(r';W\[([^\]]*)\]', content)
            
            # Interleave moves
            moves = []
            for i in range(max(len(black_moves), len(white_moves))):
                if i < len(black_moves):
                    moves.append(('B', SGFParser._sgf_to_coord(black_moves[i])))
                if i < len(white_moves):
                    moves.append(('W', SGFParser._sgf_to_coord(white_moves[i])))
            
            # Extract quality from filename if available
            quality = SGFParser._extract_quality_from_filename(file_path.name)
            if quality is not None:
                metadata['quality'] = quality
            
            return {
                'metadata': metadata,
                'moves': moves,
                'file_path': str(file_path)
            }
        except Exception as e:
            logger.error(f"Error parsing SGF file {file_path}: {e}")
            return None
    
    @staticmethod
    def _sgf_to_coord(sgf_coord):
        """Convert SGF coordinate to (row, col) tuple"""
        if not sgf_coord or len(sgf_coord) < 2 or sgf_coord.lower() == 'tt':  # Pass move
            return None
        
        col = ord(sgf_coord[0]) - ord('a')
        row = ord(sgf_coord[1]) - ord('a')
        return (row, col)
    
    @staticmethod
    def _extract_quality_from_filename(filename):
        """Extract quality value from filename if available"""
        import re
        match = re.search(r'_Q(\d+)\.sgf$', filename)
        if match:
            return int(match.group(1))
        return None


class GoBoard:
    """Simple representation of a Go board for dataset creation"""
    
    def __init__(self, size=19):
        """Initialize an empty Go board"""
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.next_player = 'B'  # Black goes first in Go
    
    def place_stone(self, color, position):
        """Place a stone on the board"""
        if position is None:  # Pass move
            self.next_player = 'W' if color == 'B' else 'B'
            return True
        
        row, col = position
        if 0 <= row < self.size and 0 <= col < self.size and self.board[row, col] == 0:
            # Place the stone (1 for black, -1 for white)
            self.board[row, col] = 1 if color == 'B' else -1
            self.next_player = 'W' if color == 'B' else 'B'
            return True
        return False
    
    def get_features(self):
        """Get features representing the current board state"""
        # Create a 3-channel representation:
        # Channel 0: Black stones
        # Channel 1: White stones
        # Channel 2: Next player (all 1s if black, all 0s if white)
        features = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Black stones
        features[0] = (self.board == 1).astype(np.float32)
        
        # White stones
        features[1] = (self.board == -1).astype(np.float32)
        
        # Next player
        features[2].fill(1.0 if self.next_player == 'B' else 0.0)
        
        return features


class GoDatasetBuilder:
    """Builder for creating a Go game dataset for neural network training"""
    
    def __init__(self, input_dir, output_dir, samples_per_category=10000, workers=4, 
                 val_split=0.1, test_split=0.1, positions_per_game=10, min_moves=30):
        """
        Initialize the dataset builder
        
        Args:
            input_dir (str): Input directory with organized SGF files
            output_dir (str): Output directory for the dataset
            samples_per_category (int): Number of samples per strength category
            workers (int): Number of worker processes
            val_split (float): Validation split ratio
            test_split (float): Test split ratio
            positions_per_game (int): Maximum number of positions to sample per game
            min_moves (int): Minimum number of moves in a game to consider it valid
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.samples_per_category = samples_per_category
        self.workers = workers
        self.val_split = val_split
        self.test_split = test_split
        self.positions_per_game = positions_per_game
        self.min_moves = min_moves
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.stats = {
            'total_sgf_files': 0,
            'valid_sgf_files': 0,
            'processed_games': 0,
            'total_positions': 0,
            'positions_by_category': {cat: 0 for cat in STRENGTH_CATEGORIES.keys()},
            'games_by_category': {cat: 0 for cat in STRENGTH_CATEGORIES.keys()},
            'processing_time': 0,
        }
    
    def find_sgf_files(self):
        """Find all SGF files in the input directory"""
        sgf_files = list(self.input_dir.rglob('*.sgf'))
        logger.info(f"Found {len(sgf_files)} SGF files in {self.input_dir}")
        self.stats['total_sgf_files'] = len(sgf_files)
        return sgf_files
    
    def group_files_by_category(self, sgf_files):
        """Group SGF files by strength category"""
        categorized_files = {category: [] for category in STRENGTH_CATEGORIES.keys()}
        
        for file_path in tqdm(sgf_files, desc="Categorizing files"):
            try:
                # Extract quality from filename
                quality = SGFParser._extract_quality_from_filename(file_path.name)
                
                if quality is None:
                    continue
                
                # Assign to category
                for category, (min_q, max_q) in STRENGTH_CATEGORIES.items():
                    if min_q <= quality <= max_q:
                        categorized_files[category].append(file_path)
                        break
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Log category statistics
        for category, files in categorized_files.items():
            self.stats['games_by_category'][category] = len(files)
            logger.info(f"Category '{category}': {len(files)} files")
        
        return categorized_files
    
    def process_game(self, file_path, category):
        """
        Process a single game from an SGF file
        
        Args:
            file_path (Path): Path to the SGF file
            category (str): Strength category
            
        Returns:
            list: List of sampled positions
        """
        try:
            # Parse SGF file
            game_data = SGFParser.parse_sgf(file_path)
            if game_data is None or len(game_data['moves']) < self.min_moves:
                return []
            
            # Sample positions from the game
            positions = self.sample_positions(game_data)
            
            # Create feature tensors
            samples = []
            for position_idx, (board_tensor, next_move, move_idx) in enumerate(positions):
                # Get the actual move played (will be our label)
                move_color, move_coord = game_data['moves'][move_idx]
                
                # Skip pass moves
                if move_coord is None:
                    continue
                
                # Convert move to 1D index (for classification)
                move_row, move_col = move_coord
                move_idx_1d = move_row * BOARD_SIZE + move_col
                
                # Create sample
                sample = {
                    'board': board_tensor,  # Board state (input features)
                    'move': move_idx_1d,    # Move played (label)
                    'color': 1 if move_color == 'B' else 0,  # Color that played the move
                    'category': category,   # Strength category
                    'game_id': os.path.basename(file_path),  # Game identifier
                    'position': position_idx  # Position index within game
                }
                samples.append(sample)
            
            return samples
        except Exception as e:
            logger.error(f"Error processing game {file_path}: {e}")
            return []
    
    def sample_positions(self, game_data):
        """
        Sample strategic positions from a game
        
        Args:
            game_data (dict): Game data from SGF parser
            
        Returns:
            list: List of (board_tensor, next_move, move_idx) tuples
        """
        moves = game_data['moves']
        total_moves = len(moves)
        
        # Skip early game and very late game
        start_idx = min(30, total_moves // 5)  # Skip first 30 moves or first 1/5 of game
        end_idx = max(0, total_moves - 30)     # Skip last 30 moves
        
        # Adjust if necessary
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = max(30, total_moves)
        
        # Select random positions
        if end_idx - start_idx <= self.positions_per_game:
            sample_indices = list(range(start_idx, end_idx))
        else:
            sample_indices = sorted(random.sample(range(start_idx, end_idx), self.positions_per_game))
        
        # Generate board states for selected positions
        positions = []
        for move_idx in sample_indices:
            board = GoBoard(BOARD_SIZE)
            
            # Play moves up to this position
            for i in range(move_idx):
                color, coord = moves[i]
                board.place_stone(color, coord)
            
            # Get the board features
            board_tensor = torch.tensor(board.get_features(), dtype=torch.float32)
            
            # Add to the list
            positions.append((board_tensor, board.next_player, move_idx))
        
        return positions
    
    def build_dataset(self):
        """Build the complete dataset"""
        start_time = datetime.now()
        
        # Find all SGF files
        sgf_files = self.find_sgf_files()
        
        # Group files by category
        categorized_files = self.group_files_by_category(sgf_files)
        
        # Determine how many files to process per category
        files_to_process = {
            category: random.sample(files, min(self.samples_per_category // self.positions_per_game, len(files)))
            for category, files in categorized_files.items()
        }
        
        # Process each category
        all_samples = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for category, files in files_to_process.items():
            logger.info(f"Processing {len(files)} files for category '{category}'")
            
            # Process files in parallel
            samples = []
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.process_game, file_path, category): file_path for file_path in files}
                
                with tqdm(total=len(futures), desc=f"Processing {category}") as pbar:
                    for future in futures:
                        try:
                            game_samples = future.result()
                            samples.extend(game_samples)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error in executor: {e}")
                            pbar.update(1)
            
            # Count processed positions
            self.stats['positions_by_category'][category] += len(samples)
            self.stats['total_positions'] += len(samples)
            
            # Shuffle samples
            random.shuffle(samples)
            
            # Split into train/val/test
            total = len(samples)
            test_idx = int(total * (1 - self.test_split))
            val_idx = int(total * (1 - self.test_split - self.val_split))
            
            all_samples['test'].extend(samples[test_idx:])
            all_samples['val'].extend(samples[val_idx:test_idx])
            all_samples['train'].extend(samples[:val_idx])
        
        # Save datasets
        for split, samples in all_samples.items():
            if not samples:
                continue
                
            # Create a dictionary of tensors
            boards = torch.stack([sample['board'] for sample in samples])
            moves = torch.tensor([sample['move'] for sample in samples], dtype=torch.long)
            colors = torch.tensor([sample['color'] for sample in samples], dtype=torch.uint8)
            categories = [sample['category'] for sample in samples]
            game_ids = [sample['game_id'] for sample in samples]
            positions = torch.tensor([sample['position'] for sample in samples], dtype=torch.int)
            
            # Save the tensors
            data_dir = self.output_dir / split
            data_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(boards, data_dir / "boards.pt")
            torch.save(moves, data_dir / "moves.pt")
            torch.save(colors, data_dir / "colors.pt")
            
            # Save metadata as JSON
            metadata = {
                'categories': categories,
                'game_ids': game_ids,
                'positions': positions.tolist(),
                'category_mapping': {i: cat for i, cat in enumerate(STRENGTH_CATEGORIES.keys())},
                'samples_count': len(samples),
                'board_size': BOARD_SIZE,
                'channels': boards.shape[1],
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Fix: Explicitly specify UTF-8 encoding when writing JSON metadata
            with open(data_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(samples)} samples to {split} dataset")
        
        # Record processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.stats['processing_time'] = processing_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        # Save stats
        self.save_stats()
        
        # Generate README
        self.generate_readme()
        
        return all_samples
    
    def save_stats(self):
        """Save processing statistics"""
        stats_file = self.output_dir / "stats.json"
        # Fix: Explicitly specify UTF-8 encoding when writing JSON stats
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics to {stats_file}")
    
    def generate_readme(self):
        """Generate a README file for the dataset"""
        readme_file = self.output_dir / "README.md"
        
        content = f"""# Go Games Dataset for PyTorch Neural Network Training

## Overview

This dataset contains Go game positions extracted from high-quality SGF files for training neural networks. The positions are organized into three strength categories based on game quality.

## Dataset Statistics

- **Total SGF Files Processed**: {self.stats['total_sgf_files']}
- **Valid SGF Files**: {self.stats['valid_sgf_files']}
- **Total Positions**: {self.stats['total_positions']}
- **Processing Time**: {self.stats['processing_time']:.2f} seconds

## Strength Categories

The dataset is divided into three strength categories:

"""
        
        for category, (min_q, max_q) in STRENGTH_CATEGORIES.items():
            positions = self.stats['positions_by_category'].get(category, 0)
            games = self.stats['games_by_category'].get(category, 0)
            content += f"- **{category.capitalize()}** (Quality {min_q}-{max_q}): {games} games, {positions} positions\n"
        
        content += """
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

This dataset was created on {datetime.now().strftime('%Y-%m-%d')}.
"""
        
        # Fix: Explicitly specify UTF-8 encoding when writing README
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Generated README file at {readme_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Go Games Dataset Builder for PyTorch Training")
    parser.add_argument("input_dir", help="Input directory with organized SGF files")
    parser.add_argument("output_dir", help="Output directory for the dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples per category")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--positions", type=int, default=10, help="Positions to sample per game")
    parser.add_argument("--min-moves", type=int, default=30, help="Minimum moves in a game")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create dataset builder
    builder = GoDatasetBuilder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        samples_per_category=args.samples,
        workers=args.workers,
        val_split=args.val,
        test_split=args.test,
        positions_per_game=args.positions,
        min_moves=args.min_moves
    )
    
    # Build dataset
    try:
        builder.build_dataset()
        logger.info("Dataset creation completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())