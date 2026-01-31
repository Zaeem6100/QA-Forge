# Utils

This folder contains utility scripts and helper functions for the QA-Forge project.

## Contents

- **data_preprocessing.py**: Scripts for cleaning and preparing datasets
- **model_evaluation.py**: Evaluation metrics and tools (EM, F1, etc.)
- **visualization.py**: Plotting training curves and model performance
- **config_loader.py**: Configuration file handling
- **dataset_converter.py**: Converting between different dataset formats

## Usage

Import utilities in your training and evaluation scripts:

```python
from utils.data_preprocessing import prepare_qa_dataset
from utils.model_evaluation import calculate_f1_score
```

## Adding New Utilities

When adding new utility functions:
1. Create descriptive function names
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests
