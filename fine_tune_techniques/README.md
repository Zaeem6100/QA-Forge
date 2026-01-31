# Fine-Tune Techniques

This folder contains implementations of different fine-tuning techniques for QA models.

## Available Techniques

### Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA** (Low-Rank Adaptation): Efficient fine-tuning by adding low-rank matrices
- **QLoRA**: Quantized LoRA for memory-efficient training
- **Adapter Layers**: Adding small adapter modules to pre-trained models
- **Prefix Tuning**: Learning continuous task-specific vectors

### Full Fine-Tuning
- **Standard Fine-Tuning**: Full model parameter updates
- **Gradual Unfreezing**: Progressive layer unfreezing during training

## Structure

Each technique should have its own directory containing:
- Implementation scripts
- Configuration files
- Training pipelines
- Evaluation scripts

## Usage

Choose a fine-tuning technique based on:
- Available computational resources
- Model size
- Dataset size
- Target performance requirements
