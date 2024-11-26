#!/bin/sh

echo "for llama : python train.py --model_id \"meta-llama/Llama-2-3.2b-hf\""

echo '.venv/bin/python train.py --output_dir="./my_training_run" --resume_from_checkpoint="./my_training_run/checkpoint-1000"'


#.venv/bin/python train.py
.venv/bin/python train.py --push_to_hub False --output_dir /Users/seandolbec/Downloads/smollm/finetuning/finetune_smollm2_python


