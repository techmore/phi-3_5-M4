# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import multiprocessing
import os
import time
import psutil
from datetime import datetime

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
import gc
import os
import psutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    #parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--subset", type=str, default="data/python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="instruction")

    # Training settings optimized for Phi on Apple Silicon
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=8)  # Back to proven setting
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)  # Back to proven setting
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)  # Back to proven setting
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--torch_dtype", type=str, default="float32")  # Use float32 for MPS compatibility
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_smollm2_python")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--save_merged_model", type=bool, default=True)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--repo_id", type=str, default="SmolLM2-1.7B-finetune")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
    return parser.parse_args()


def format_throughput(num):
    return f"{num:,.2f}" if num < 1000 else f"{num:,.2f}k"


class PerformanceCallback(transformers.TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.total_examples = 0

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        step_time = current_time - self.step_start_time
        examples_per_step = args.gradient_accumulation_steps * args.per_device_train_batch_size
        self.total_examples += examples_per_step
        
        # Calculate throughput
        examples_per_second = examples_per_step / step_time
        avg_examples_per_second = self.total_examples / (current_time - self.start_time)
        
        # Get memory usage
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        
        # Calculate ETA
        elapsed_time = current_time - self.start_time
        progress = state.global_step / state.max_steps
        if progress > 0:
            eta_seconds = (elapsed_time / progress) * (1 - progress)
            eta = datetime.fromtimestamp(current_time + eta_seconds).strftime('%H:%M:%S')
        else:
            eta = "calculating..."

        print(f"Speed: {format_throughput(examples_per_second)} examples/s (avg: {format_throughput(avg_examples_per_second)}/s) | "
              f"Memory: {memory_gb:.1f}GB | ETA: {eta}")
        
        self.step_start_time = current_time


def print_memory_usage():
    # Convert to GB for readability
    process = psutil.Process()
    print(f"RAM Memory Usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def main(args):
    # Initial memory cleanup
    clear_memory()
    print("Initial memory usage:")
    print_memory_usage()

    # Set up device and handle MPS specifically
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16 if args.fp16 else torch.float32
        # Configure quantization for CUDA only
        compute_dtype = torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS doesn't fully support float16 yet
        torch.backends.mps.enable_fallback_to_cpu = True
        quantization_config = None
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        quantization_config = None
    print(f"Using device: {device} with dtype: {dtype}")

    # Memory optimization settings
    model_kwargs = {
        "device_map": None if device.type == "mps" else "auto",  # Don't use device_map for MPS
        "use_cache": False,
        "attention_dropout": args.attention_dropout,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",  # Disable flash attention warnings
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    print("Before loading dataset:")
    print_memory_usage()
    
    # Load dataset with memory efficient settings
    try:
        # Load dataset in streaming mode
        data = load_dataset(
            'json',
            data_files={'train': 'training_data.json'},
            split='train'  # Remove streaming=True to get proper Dataset object
        )
        
        # Select a subset for memory efficiency
        if len(data) > 500:
            data = data.select(range(500))
            
        print("Dataset sample:", data[0] if len(data) > 0 else None)
        print("Dataset size:", len(data))
        
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}\nMake sure training_data.json exists in the current directory.")

    print("Before loading model:")
    print_memory_usage()
    clear_memory()

    # Load model with optimized settings
    print("Loading model on mps...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
            device_map={"": device}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
        # Ensure model is in training mode
        model.train()
        # Enable gradient computation
        for param in model.parameters():
            param.requires_grad = True
            
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

    print("After loading model:")
    print_memory_usage()
    clear_memory()

    # Setup LoRA config with optimized settings
    peft_config = LoraConfig(
        r=24,
        lora_alpha=48,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA config to model
    print("Applying LoRA config...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Before initializing trainer:")
    print_memory_usage()
    clear_memory()

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        tokenizer=tokenizer,
        dataset_text_field="instruction",
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=24,
            gradient_accumulation_steps=3,
            warmup_steps=100,
            max_steps=2000,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            fp16=False,
            bf16=False,
            torch_compile=False,
            optim="adamw_torch",
            gradient_checkpointing=False,
            logging_strategy="steps",
            logging_steps=10,
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to="none",
            # Memory and performance settings
            dataloader_num_workers=0,  # Reverting to single-threaded loading for stability
            dataloader_pin_memory=True,  # Keep pinned memory for efficient transfers
            max_grad_norm=1.0,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            ddp_find_unused_parameters=False,
            eval_strategy="no",
            load_best_model_at_end=False,
            hub_strategy="end",
            hub_model_id=None,
            hub_token=None,
            push_to_hub=False,
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        callbacks=[PerformanceCallback()],
    )

    # Add memory cleanup callback
    class MemoryCleanupCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 5 == 0:  # Clean more frequently
                clear_memory()
                print(f"Step {state.global_step} memory usage:")
                print_memory_usage()

    trainer.add_callback(MemoryCleanupCallback())

    print("Starting training...")
    print_memory_usage()
    trainer.train()

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))

    if args.save_merged_model:
        print("Saving merged model...")
        # Free memory for merging weights
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            device_map=device,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16
        )
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)

        if args.push_to_hub:
            model.push_to_hub(args.repo_id, "Upload model")
    
    print("Training Done! ")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)