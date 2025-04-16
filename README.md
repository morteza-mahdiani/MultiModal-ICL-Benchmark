# In-Context Learning for Fine-Grained Visual Understanding

A comprehensive benchmark for evaluating multimodal models using in-context learning (ICL) across various vision-language tasks.

## Overview

This repository contains evaluation pipelines for testing different multimodal models (GPT-4V, LLaVA, IDEFICS) on various vision-language tasks. The project focuses on few-shot learning capabilities and includes implementations for both Winoground and Vismin benchmark datasets.

## Models Supported

- **GPT-4V (GPT-4 Vision)**: OpenAI's multimodal model
- **LLaVA**: Large Language and Vision Assistant
- **IDEFICS**: Image-aware Decoder Enhanced to Flow In-Context with Support

## Features

- Few-shot learning evaluation pipelines
- Support for multiple multimodal benchmarks:
  - Winoground
  - Vismin
- Cost tracking for API-based models
- Batch processing capabilities
- Flexible prompt construction
- Image processing utilities

## Project Structure

```
.
├── gpt4_fewshot_pipeline.py          # GPT-4V evaluation pipeline
├── gpt4_fewshot_winoground_pipeline.py  # GPT-4V Winoground-specific pipeline
├── LlaVa_fewshot_pipeline.py         # LLaVA evaluation pipeline
├── LlaVa_fewshot_winoground_pipeline.py  # LLaVA Winoground-specific pipeline
├── idefics_fewshot_pipeline.py       # IDEFICS evaluation pipeline
├── Winoground_Idefics.py            # IDEFICS Winoground-specific pipeline
├── gpt_main_pipeline.py             # Main GPT-4V pipeline
├── evaluate_gpt4o_winoground.py     # Winoground evaluation script
├── data/                            # Evaluation results
├── meta/                            # Some useful materials
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MultiModal-ICL-Benchmark.git
cd MultiModal-ICL-Benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

## Usage

### GPT-4V Evaluation

```bash
python gpt4_fewshot_pipeline.py --api_key YOUR_API_KEY --batch_size 8 --num_fewshot_examples 8
```

### LLaVA Evaluation

```bash
python LlaVa_fewshot_pipeline.py --batch_size 1 --num_fewshot_examples 8
```

### Winoground Evaluation

```bash
python evaluate_gpt4o_winoground.py --api_key YOUR_API_KEY
```

## Parameters

- `batch_size`: Number of samples to process in each batch
- `num_fewshot_examples`: Number of few-shot examples to use
- `use_image_question`: Whether to use image-based questions (default: True)
- `filename`: Output filename for results
- `subset_size`: Limit evaluation to a subset of the dataset
- `api_key`: API key for model access

## Results

Evaluation results are saved in CSV format in the `data/` directory. The results include:
- Sample IDs
- Categories
- Model predictions
- Ground truth labels
- Performance metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bibtex
@misc{multimodal-icl-benchmark,
  author = {Alireza Farashah, Morteza Mahdiani},
  title = {MultiModal-ICL-Benchmark},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/MultiModal-ICL-Benchmark}}
}
```
