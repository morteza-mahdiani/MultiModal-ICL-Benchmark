import os
import random
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Set API keys
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

# Load the VisMin benchmark dataset from Hugging Face
dataset = load_dataset('mair-lab/vismin-bench')
vismin_data = dataset['test']  # Using 'test' split for evaluation

# Convert dataset to a list of dictionaries for easier processing
data_samples = [sample for sample in vismin_data]

# Function to get category of a sample
def get_category(sample):
    return sample.get('category', 'default')

# Select few-shot examples from the same category (excluding the current sample)
def get_few_shot_examples(current_sample, all_samples, num_examples=8):
    current_cat = get_category(current_sample)
    same_cat = [s for s in all_samples if get_category(s) == current_cat and s != current_sample]
    few_shots = random.sample(same_cat, min(num_examples, len(same_cat)))
    return few_shots

# Build a text prompt for in-context learning
def build_text_prompt(few_shot_examples, eval_sample):
    prompt = "Below are examples of how to answer questions based on captions:\n\n"
    for idx, ex in enumerate(few_shot_examples, start=1):
        example_prompt = (
            f"Example {idx}:\n"
            f"Captions: {ex['text_0']} | {ex['text_1']}\n"
            f"Question: {ex['text_question_0']}\n"
            f"Answer: {ex.get('text_answer_0', '<<example answer>>')}\n\n"
        )
        prompt += example_prompt

    prompt += "Now, answer the following for the evaluation sample:\n"
    prompt += f"Captions: {eval_sample['text_0']} | {eval_sample['text_1']}\n"
    prompt += f"Question: {eval_sample['text_question_0']}\n"
    prompt += "Answer: "
    return prompt

# Evaluate text-based questions using OpenAI API
def evaluate_text_question(prompt):
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    return response.choices[0].message.content

# Evaluate image-based questions using OpenAI Vision API
def evaluate_image_question(image_path, question):
    try:
        response = client.images.generate(file=open(image_path, 'rb'),
        prompt=question,
        n=1,
        size="1024x1024")
        return response
    except Exception as e:
        return f"Error processing image: {e}"

# Main evaluation loop
results = []
for sample in data_samples:
    few_shot = get_few_shot_examples(sample, data_samples, num_examples=8)
    text_prompt = build_text_prompt(few_shot, sample)
    text_answer = evaluate_text_question(text_prompt)

    # Handle image question evaluation
    try:
        image_eval = evaluate_image_question(sample['image_0'], sample['image_question_0'])
    except Exception as e:
        image_eval = f"Error processing image: {e}"

    results.append({
        'sample_id': sample.get('id', None),
        'category': get_category(sample),
        'text_prompt': text_prompt,
        'text_answer': text_answer,
        'image_question_0_response': image_eval
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('multimodal_icl_benchmark_results.csv', index=False)

print("Evaluation complete. Results saved to multimodal_icl_benchmark_results.csv")

