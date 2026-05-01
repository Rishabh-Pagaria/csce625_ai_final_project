# scripts/evaluate_gemma.py

import argparse
import sys
import os
from pathlib import Path

# Disable torch dynamo BEFORE importing torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def evaluate_model(model, tokenizer, dataset, device, max_samples=None):
    """
    Evaluate the model on the given dataset.
    Returns accuracy, precision, recall, F1, and confusion matrix.
    """
    model.eval()
    predictions = []
    labels = []
    
    # Limit samples if specified
    eval_data = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(eval_data)} samples...")
    
    for sample in tqdm(eval_data, desc="Evaluating"):
        # Create the conversation in the EXACT format used during training
        messages = [
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
            {"role": "assistant", "content": ""}  # Empty for generation
        ]
        
        # Apply chat template (same as training)
        prompt = tokenizer.apply_chat_template(
            messages[:-1],  # Only user message, not assistant
            tokenize=False,
            add_generation_prompt=True  # Adds the assistant prompt
        )
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,   # Just need "phishing" or "safe"
                do_sample=False,    # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode ONLY the new tokens (not the prompt)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        
        # Map to label (0=safe, 1=phishing)
        if "phishing" in prediction_text:
            pred_label = 1
        elif "safe" in prediction_text:
            pred_label = 0
        else:
            # If unclear, default to safe
            pred_label = 0
            if len(prediction_text) > 0:
                print(f"Warning: Unclear prediction '{prediction_text[:50]}' - defaulting to 'safe'")
        
        predictions.append(pred_label)
        labels.append(1 if sample['output'].lower() == "phishing" else 0)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma-2-2b model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="artifacts/models/gemma-2-2b-it-phishing",
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="hf_models/gemma-2-2b-it",
        help="Path to base Gemma model",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="data/eval_gemma.jsonl",
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test_gemma.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/models/gemma-2-2b-phishing/evaluation_results.json",
        help="Where to save evaluation results (JSON)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="artifacts/result.csv",
        help="Where to save evaluation results (CSV)",
    )
    args = parser.parse_args()

    print(f"[evaluate_gemma] Starting evaluation...")
    print(f"[evaluate_gemma] Model: {args.model_path}")
    print(f"[evaluate_gemma] Eval file: {args.eval_file}")
    print(f"[evaluate_gemma] Test file: {args.test_file}")

    # Configure quantization if using QLoRA
    if args.use_qlora:
        print("[evaluate_gemma] Using QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Load tokenizer
    print("[evaluate_gemma] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    print("[evaluate_gemma] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter and merge weights
    print(f"[evaluate_gemma] Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(model, args.model_path)
    model = model.merge_and_unload()  # Critical: merge LoRA weights into base model
    model.eval()
    
    device = next(model.parameters()).device
    print(f"[evaluate_gemma] Model loaded on device: {device}")

    # Load datasets
    print("[evaluate_gemma] Loading datasets...")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    test_dataset = load_dataset("json", data_files=args.test_file, split="train")

    # Evaluate on validation set
    print("\n" + "="*50)
    print("EVALUATION SET RESULTS")
    print("="*50)
    eval_results = evaluate_model(model, tokenizer, eval_dataset, device, args.max_samples)
    
    print(f"\nAccuracy:  {eval_results['accuracy']:.4f}")
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall:    {eval_results['recall']:.4f}")
    print(f"F1 Score:  {eval_results['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={eval_results['confusion_matrix'][0][0]}, FP={eval_results['confusion_matrix'][0][1]}],")
    print(f"   [FN={eval_results['confusion_matrix'][1][0]}, TP={eval_results['confusion_matrix'][1][1]}]]")

    # Evaluate on test set
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    test_results = evaluate_model(model, tokenizer, test_dataset, device, args.max_samples)
    
    print(f"\nAccuracy:  {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall:    {test_results['recall']:.4f}")
    print(f"F1 Score:  {test_results['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={test_results['confusion_matrix'][0][0]}, FP={test_results['confusion_matrix'][0][1]}],")
    print(f"   [FN={test_results['confusion_matrix'][1][0]}, TP={test_results['confusion_matrix'][1][1]}]]")

    # Save results as JSON (original behavior)
    results = {
        "model_path": args.model_path,
        "eval_set": eval_results,
        "test_set": test_results,
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate_gemma] Results saved to {args.output_file}")

    # Save results as CSV (new behavior)
    import csv
    csv_path = args.csv_file
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Set", "Accuracy", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP", "Num Samples"])
        for set_name, res in zip(["eval", "test"], [eval_results, test_results]):
            cm = res["confusion_matrix"]
            tn, fp = cm[0]
            fn, tp = cm[1]
            writer.writerow([
                set_name,
                f"{res['accuracy']:.4f}",
                f"{res['precision']:.4f}",
                f"{res['recall']:.4f}",
                f"{res['f1']:.4f}",
                tn, fp, fn, tp, res["num_samples"]
            ])
    print(f"[evaluate_gemma] CSV results saved to {csv_path}")


if __name__ == "__main__":
    main()
