# Models

This folder contains fine-tuned QA models and their configurations.

## Structure

- **checkpoints/**: Training checkpoints saved during fine-tuning
- **final/**: Final trained models ready for inference
- **configs/**: Model configuration files

## Model Format

Models are saved in formats compatible with popular frameworks:
- PyTorch (.pt, .pth)
- Hugging Face Transformers
- ONNX (for deployment)

## Usage

After fine-tuning, save your models here with descriptive names including:
- Base model name
- Fine-tuning technique used
- Dataset used
- Version/date

Example: `bert-base-qa-lora-squad-v1.0`
