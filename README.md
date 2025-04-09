# Chain-of-Thought Mathematical Reasoning with Qwen 3B

## Overview
This repository demonstrates fine-tuning the Qwen 3B model to perform mathematical reasoning using the Chain-of-Thought (CoT) prompting strategy. Leveraging advanced training methodologies and efficient fine-tuning strategies, this implementation aims to enhance the Qwen model's reasoning and problem-solving capabilities.

## Features
- **Model Architecture**: Qwen2.5-3B fine-tuned with LoRA (Low-Rank Adaptation).
- **Prompt Strategy**: Structured Chain-of-Thought prompting to improve reasoning clarity.
- **Fine-tuning Method**: Efficient Parameter-Efficient Fine-Tuning (PEFT) using the `unsloth` framework.
- **Dataset**: Mathematical reasoning tasks extracted from the "Ashkchamp/Openthoughts_math_filtered_30K" dataset.

## Technologies Used
- **Unsloth**: For rapid and memory-efficient fine-tuning of LLM models.
- **Hugging Face Transformers & TRL**: Leveraged for model manipulation, training, and evaluation.
- **PEFT (LoRA)**: Enables efficient fine-tuning by adapting only select model parameters.

## Dataset Preparation
Dataset used:
- [Ashkchamp/Openthoughts_math_filtered_30K](https://huggingface.co/datasets/Ashkchamp/Openthoughts_math_filtered_30K)

Data preprocessing involves structured prompts containing the system context, mathematical questions, detailed Chain-of-Thought reasoning, and explicitly formatted solutions.

## Training
Fine-tuning configurations:
- Batch Size: 2 (with gradient accumulation of 4 steps)
- Sequence Length: 8192 tokens
- Learning Rate: 2e-5
- Optimizer: AdamW (8-bit)
- Precision: Automatic FP16/BF16 based on hardware support
- LoRA Rank: 32

## Usage
Post-training, the model is equipped to handle mathematical queries employing Chain-of-Thought reasoning clearly and systematically:

## Contribution
Feel free to contribute improvements, report issues, or suggest new features via pull requests or issue submissions.

## License
Distributed under the MIT License. See the `LICENSE` file for more information.
