# Go Game Dataset Construction Project Documentation

## Project Overview

In the era of rapid advancement in artificial intelligence and machine learning, high-quality training datasets are crucial for the performance of deep learning models. Go, as an extremely complex strategic board game, presents unique challenges in dataset construction. This project meticulously builds a high-quality, structured Go game dataset from massive and chaotic SGF (Smart Game Format) files, providing a solid foundation for deep learning research.

## Hugging Face Dataset

This dataset is now available on Hugging Face Datasets Hub:

🤗 **Dataset Link**: https://huggingface.co/datasets/Karesis/GoDatas

### Loading with Hugging Face Datasets

You can easily load this dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Karesis/GoDatas")

# Access specific splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Example: Get a board state and its corresponding move
board = train_data[0]["board_state"]
move = train_data[0]["move"]

### Data Source
Base dataset sourced from: https://homepages.cwi.nl/~aeb/go/games

The raw data collection represents a comprehensive compilation of Go games, requiring sophisticated processing to transform raw SGF files into a structured, machine-learning-ready format.

## Key Components

The project consists of two core Python scripts:

1. `gogames_organizer.py`: Dataset Organizing Tool
2. `go_training_dataset.py`: Dataset Builder

### gogames_organizer.py

#### Functionality
- Recursively find and process SGF files
- Repair and standardize SGF file metadata
- Evaluate game quality based on predefined rules
- Support multiple file organization methods (by quality, player, date, event)
- Remove duplicate files
- Generate processing statistical reports

#### Key Features
- Built-in professional player and tournament database
- Multi-threaded parallel processing
- Flexible metadata extraction and repair mechanism
- Automatic HTML statistical report generation

### go_training_dataset.py

#### Functionality
- Build PyTorch training dataset from organized SGF files
- Convert game records to standardized tensor representations
- Support multi-process data processing
- Generate training, validation, and test datasets

#### Data Representation
- 3-channel feature representation:
  1. Black stone positions
  2. White stone positions
  3. Next move turn

## Dataset Construction Workflow

1. Use `gogames_organizer.py` to organize raw SGF files
   ```bash
   python gogames_organizer.py input_dir output_dir --quality 80 --organize quality
   ```

2. Use `go_training_dataset.py` to build training dataset
   ```bash
   python go_training_dataset.py input_sgf_dir output_dataset_dir
   ```

## Dataset Characteristics

- 19x19 board size
- Three strength levels:
  - Standard (Quality 80-85)
  - Strong (Quality 86-92)
  - Elite (Quality 93-100)
- Maximum 10 positions sampled per game
- Dataset includes:
  - `boards.pt`: Board state tensors
  - `moves.pt`: Actual move positions
  - `colors.pt`: Move colors
  - `metadata.json`: Dataset metadata

## Code Highlights

- Modular design
- Highly configurable
- Parallel processing support
- Detailed logging
- Automatic metadata repair and standardization

## Performance and Scale

- Supports large-scale SGF file processing
- Multi-threaded data processing acceleration
- Flexible quality filtering mechanism

## Usage Recommendations

1. Ensure input SGF file directory structure is clear
2. Adjust parallel worker threads based on computational resources
3. Customize dataset generation strategy via command-line parameters

## Dependencies

- Python 3.7+
- NumPy
- PyTorch
- Tqdm
- Papaparse (optional)

## License

This project is intended for research and educational purposes only.

## Contribution and Feedback

Suggestions for improvements and code contributions are welcome through Issues and Pull Requests.

## Authors

[Your Name]

## Version History

- v1.0: Initial release
- v1.1: Performance optimization and bug fixes

## Technical Details

### Metadata Enhancement

The project implements sophisticated metadata enhancement techniques:

- Player name standardization
- Date format normalization
- Tournament name mapping
- Quality scoring algorithm

### Data Sampling Strategy

- Skip initial and final game moves
- Randomly sample strategic positions
- Ensure diverse representation of game stages

### Quality Assessment Criteria

Game quality is evaluated based on multiple factors:
- Player professional ranks
- Tournament prestige
- Player reputation
- Recent game annotations
- Temporal relevance

## Future Improvements

- Support for additional board sizes
- Enhanced machine learning model compatibility
- More comprehensive metadata analysis
- Expanded player and tournament databases

## Performance Metrics

- Processing Speed: Optimized for large datasets
- Memory Efficiency: Minimal memory footprint
- Scalability: Supports extensive SGF collections

## Research Potential

This dataset is particularly valuable for:
- Move prediction models
- Game strategy analysis
- AI training in Go
- Historical game pattern recognition