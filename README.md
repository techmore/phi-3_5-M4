# Phi Model Fine-tuning

This repository provides scripts for fine-tuning the Microsoft Phi model using PEFT (Parameter Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation) and the TRL (Transformer Reinforcement Learning) library. The setup is optimized for running on Apple Silicon.

## Setup

Install `pytorch` [see documentation](https://pytorch.org/), and then install the requirements:
```bash
pip install -r requirements.txt
```

Before running any scripts, configure your environment:
```bash
huggingface-cli login  # For model access
accelerate config  # For training configuration
```

## Training

The training script uses PEFT for efficient fine-tuning with LoRA and the `SFTTrainer` from TRL. Basic usage:

```bash
python train.py \
    --model_id "microsoft/Phi-3.5-mini-instruct" \
    --output_dir "./my_training_run"
```

Default training parameters are optimized for Phi on Apple Silicon:
- Max sequence length: 512
- Micro batch size: 8
- Gradient accumulation steps: 16
- Learning rate: 1e-4
- BF16 training: enabled
- Gradient checkpointing: enabled
- Float32 dtype for MPS compatibility

The training uses a JSON dataset with the following fields:
- `instruction`: The main text field used for training
- `input`: Optional context or input
- `output`: Expected output

## Training Monitoring & Checkpoints

### Checkpoint System
- Checkpoints are saved every 500 training steps
- Only the last 2 checkpoints are kept to manage storage
- Checkpoints are saved in your specified `output_dir`
- Resume training using `--resume_from_checkpoint="path/to/checkpoint"`

### Training Progress
- Training logs are output every 10 steps
- Default maximum training steps: 2000
- Memory usage monitoring every 5 steps
- Memory optimization features:
  - Gradient checkpointing
  - Memory cleanup after steps
  - 4-bit quantization for CUDA devices
  - MPS (Apple Silicon) optimizations

### Testing Your Model
The repository includes a `test_model.py` script for evaluating your trained model:
```bash
python test_model.py
```

The test script:
- Loads your trained model from the checkpoint directory
- Runs test prompts through the model
- Supports both CPU and MPS (Apple Silicon GPU)
- Uses temperature=0.7 for generation diversity

### Potential Improvements
The current training setup focuses on training loss. For more comprehensive evaluation:

1. Add Evaluation Metrics:
   - Implement BLEU, ROUGE, or custom accuracy metrics
   - Add validation datasets
   - Create evaluation callbacks

2. Enhanced Monitoring:
   - Add validation sets for periodic evaluation
   - Implement custom metric tracking
   - Monitor training convergence

3. Custom Evaluation:
   - Create domain-specific test sets
   - Add automated benchmarking
   - Track specific performance metrics
