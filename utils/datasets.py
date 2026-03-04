"""
Dataset utilities for mamba-spectral experiments.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_associative_recall(
    n_samples: int = 100,
    key_length: int = 3,
    distance: int = 50,
    vocab_size: int = 100,
    distractor_tokens: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate associative recall dataset.
    
    Creates key-value pairs with distractors between the key
    definition and the query.
    
    Format: "K=[key]. [distractors...] Q: K?"
    Answer: [key]
    
    Args:
        n_samples: Number of samples to generate.
        key_length: Length of key/value in tokens.
        distance: Number of distractor tokens.
        vocab_size: Size of vocabulary for tokens.
        distractor_tokens: Tokens per distractor word.
        
    Returns:
        List of dicts with 'prompt' and 'answer' keys.
        
    Example:
        >>> data = generate_associative_recall(n_samples=10, distance=20)
        >>> print(data[0]['prompt'])
        'K=ABC. [distraction...] Q: K?'
    """
    samples = []
    
    for _ in range(n_samples):
        # Generate key
        key = ''.join(chr(65 + random.randint(0, 25)) for _ in range(key_length))
        
        # Generate distractors
        n_distractors = distance // distractor_tokens
        distractors = []
        for _ in range(n_distractors):
            word = ''.join(chr(97 + random.randint(0, 25)) for _ in range(distractor_tokens))
            distractors.append(word)
        distractor_text = ' '.join(distractors)
        
        # Build prompt
        prompt = f"K={key}. {distractor_text} Q: K?"
        
        samples.append({
            "prompt": prompt,
            "answer": key,
            "distance": distance,
            "key_length": key_length,
        })
    
    return samples


def generate_math_problems(
    n_samples: int = 100,
    difficulty: str = "easy",
    cot_style: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate arithmetic problems for reasoning tests.
    
    Args:
        n_samples: Number of problems.
        difficulty: 'easy', 'medium', or 'hard'.
        cot_style: Include chain-of-thought prompting.
        
    Returns:
        List of dicts with 'prompt' and 'answer' keys.
    """
    samples = []
    
    if difficulty == "easy":
        max_val = 100
        ops = ['+', '-']
        n_ops = 1
    elif difficulty == "medium":
        max_val = 1000
        ops = ['+', '-', '*']
        n_ops = 2
    else:  # hard
        max_val = 10000
        ops = ['+', '-', '*', '/']
        n_ops = 3
    
    for _ in range(n_samples):
        # Generate expression
        nums = [random.randint(1, max_val) for _ in range(n_ops + 1)]
        chosen_ops = [random.choice(ops) for _ in range(n_ops)]
        
        # Build expression string
        expr_parts = [str(nums[0])]
        for i, op in enumerate(chosen_ops):
            expr_parts.append(op)
            expr_parts.append(str(nums[i + 1]))
        expr = ' '.join(expr_parts)
        
        # Compute answer
        try:
            answer = eval(expr)
            if isinstance(answer, float):
                answer = round(answer, 2)
        except Exception:
            answer = 0
        
        if cot_style:
            prompt = f"Let's solve step by step: {expr} = ?"
        else:
            prompt = f"Compute: {expr} = ?"
        
        samples.append({
            "prompt": prompt,
            "answer": str(answer),
            "expression": expr,
            "difficulty": difficulty,
        })
    
    return samples


def load_safe_prompts(n: int = 100) -> List[str]:
    """
    Load a collection of known-safe prompts.
    
    These are benign prompts used to calibrate SpectralGuard
    and establish baseline spectral behavior.
    
    Args:
        n: Number of prompts to return.
        
    Returns:
        List of safe prompt strings.
    """
    # Collection of safe, general-purpose prompts
    base_prompts = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a haiku about autumn.",
        "What are the primary colors?",
        "How do airplanes fly?",
        "Describe the water cycle.",
        "What is the speed of light?",
        "Who invented the telephone?",
        "Explain the theory of relativity.",
        "What causes rainbows?",
        "How does a computer work?",
        "What is DNA?",
        "Explain machine learning basics.",
        "What is the largest planet?",
        "How do vaccines work?",
        "What is climate change?",
        "Explain the Big Bang theory.",
        "What are black holes?",
        "How do magnets work?",
        "What is electricity?",
        "Explain how the internet works.",
        "What is artificial intelligence?",
        "How do plants grow?",
        "What causes earthquakes?",
        "Explain the greenhouse effect.",
        "What is quantum computing?",
        "How do batteries work?",
        "What is gravity?",
        "Explain nuclear fusion.",
        "What are vitamins?",
        "How do submarines work?",
        "What is evolution?",
        "Explain blockchain technology.",
        "What causes thunder?",
        "How do telescopes work?",
        "What is the immune system?",
        "Explain renewable energy.",
        "What are neurons?",
        "How do satellites orbit?",
        "What is the periodic table?",
    ]
    
    # Extend with variations if needed
    prompts = []
    while len(prompts) < n:
        for base in base_prompts:
            if len(prompts) >= n:
                break
            # Add variations
            variations = [
                base,
                f"Please {base.lower()}",
                f"Can you {base.lower()}",
                f"Help me understand: {base}",
            ]
            prompts.append(random.choice(variations))
    
    return prompts[:n]


def generate_adversarial_samples(
    n_samples: int = 50,
    attack_type: str = "random",
) -> List[Dict[str, Any]]:
    """
    Generate adversarial-style samples for testing.
    
    Args:
        n_samples: Number of samples.
        attack_type: Type of adversarial pattern.
        
    Returns:
        List of adversarial sample dicts.
    """
    samples = []
    
    for i in range(n_samples):
        if attack_type == "random":
            # Random noise injection
            noise = ''.join(chr(random.randint(33, 126)) for _ in range(50))
            prompt = f"Normal text {noise} more text"
        elif attack_type == "padding":
            # Excessive padding
            prompt = "Question" + "   " * 100 + "Answer?"
        elif attack_type == "repetition":
            # Repetitive patterns
            prompt = "test " * 200
        else:
            prompt = f"Sample {i}"
        
        samples.append({
            "prompt": prompt,
            "type": attack_type,
            "is_adversarial": True,
        })
    
    return samples
