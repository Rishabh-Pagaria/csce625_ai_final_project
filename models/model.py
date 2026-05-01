"""
Gemma 2-2b model initialization and helper functions for text generation.
This module provides a centralized model instance that can be imported by other modules.
"""

import torch
from transformers import pipeline
from typing import Dict

# Initialize the model pipeline (using 2b version for faster inference)
pipe = pipeline(
    task="text-generation",
    model="google/gemma-2-2b-it",
    dtype=torch.bfloat16,
    device_map="auto",
)

def generate_phishing_analysis(
    email_text: str,
    max_length: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Generate phishing classification using Gemma 2-2b model.
    
    Args:
        email_text (str): The email text to analyze
        max_length (int): Maximum length of generated response
        temperature (float): Sampling temperature (0.0-1.0)
        top_p (float): Nucleus sampling parameter
        
    Returns:
        str: Classification result ("phishing" or "safe")
    """

    # Updated prompt to request explanation
    prompt = f"""You are a security model. Classify the following email as 'phishing' or 'safe' and provide a detailed explanation for your decision.

Email:
{email_text}

Your response should start with the classification ('phishing' or 'safe'), followed by a colon and then the explanation.
Example: phishing: This email contains suspicious links and urgent language.
"""


    # Generate response
    response = pipe(
        prompt,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        return_full_text=False,
    )[0]["generated_text"]

    # Extract classification and explanation
    if ":" in response:
        classification, explanation = response.split(":", 1)
        classification = classification.strip().lower()
        explanation = explanation.strip()
    else:
        classification = response.strip().lower().split()[0] if response.strip() else "unknown"
        explanation = response.strip()
    return {"classification": classification, "explanation": explanation}

# Singleton instance that can be imported by other modules
model = pipe
