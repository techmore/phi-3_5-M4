from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path="finetune_smollm2_python/final_merged_checkpoint"):
    # Load the merged model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Move to MPS (Apple Silicon GPU) if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, device, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load the model
    model, tokenizer, device = load_model()
    
    # Test prompts
    test_prompts = [
        "Write a Python function to calculate the fibonacci sequence",
        "Write a function to sort a list in Python",
        "How do I read a file in Python?"
    ]
    
    print("\nTesting the model with different prompts:\n")
    for prompt in test_prompts:
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print("Response:")
        response = generate_text(model, tokenizer, prompt, device)
        print(response)
        print()
