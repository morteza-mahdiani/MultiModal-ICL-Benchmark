from typing import Union, List
import PIL
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextVideoForConditionalGeneration, LlavaForConditionalGeneration
import random
import pandas as pd
from datasets import load_dataset
import os
import argparse
from tqdm import tqdm
import math


class LlavaModel:
    def __init__(self, model_name_or_path="llava-hf/llava-interleave-qwen-7b-hf", device="cuda", **kwargs):
        self.device = device
        cache_dir = "./huggingface_cache"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def prepare_prompt(self, text: str, images: List[PIL.Image.Image]):
        content = [{"type": "text", "text": text}]
        for img in images:
            content.append({"type": "image", "image": img})
        message = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
        return prompt

    def predict(self, text: str, images: Union[str, PIL.Image.Image, List[str], List[PIL.Image.Image]], max_new_tokens=64):
        # Normalize image input
        if isinstance(images, str):
            images = [PIL.Image.open(images).convert("RGB")]
        elif isinstance(images, PIL.Image.Image):
            images = [images]
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [PIL.Image.open(img).convert("RGB") for img in images]

        # Prepare prompt and inputs
        prompt = self.prepare_prompt(text, images)
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device, torch.float16)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False
        )

        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def predict_batch(self, texts: List[str], image_batches: List[List[Union[str, PIL.Image.Image]]], max_new_tokens=64):
        outputs = []
        for text, images in zip(texts, image_batches):
            norm_images = []
            for img in images:
                if isinstance(img, str):
                    norm_images.append(PIL.Image.open(img).convert("RGB"))
                else:
                    norm_images.append(img)
            output = self.predict(text, norm_images, max_new_tokens=max_new_tokens)
            outputs.append(output)
        return outputs

def get_category(sample):
    return sample.get('category', 'default')

def get_few_shot_examples(current_sample, all_samples, num_examples=8):
    current_cat = get_category(current_sample)
    same_cat = [s for s in all_samples if get_category(s) == current_cat and s != current_sample]
    return random.sample(same_cat, min(num_examples, len(same_cat)))

def build_few_shot_prompt_and_images(few_shot_examples, eval_sample):
    prompt = "Below are examples selected from the same category as the evaluation sample. Carefully study the question and answer for each.\n\n"
    images = []

    for idx, ex in enumerate(few_shot_examples, start=1):
        if random.random() < 0.5:
            image_field = "image_0"
            q_field = "text_question_0"
            a_field = "text_answer_0"
            default_answer = "A"
        else:
            image_field = "image_1"
            q_field = "text_question_1"
            a_field = "text_answer_1"
            default_answer = "B"

        question_text = ex[q_field].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
        example_text = (
            f"Example {idx}:\n"
            f"Question: {question_text}\n"
            f"Answer: {ex.get(a_field, default_answer)}\n\n"
        )
        prompt += example_text
        images.append(ex[image_field])
    
    eval_question_text = eval_sample['text_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_prompt = "Now, based on the samples above, please answer this evaluation question. Explain your reasoning before giving the final answer:\n"
    prompt_1 = prompt + eval_prompt + f"Question: {eval_question_text}\nAnswer: "

    images_1 = images.copy()
    images_1.append(eval_sample['image_0'])

    eval_question_text = eval_sample['text_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    prompt_2 = prompt + eval_prompt + f"Question: {eval_question_text}\nAnswer: "
    
    images_2 = images.copy()
    images_2.append(eval_sample['image_1'])

    return prompt_1, images_1, prompt_2, images_2

def build_few_shot_prompt_and_images_for_image_question(few_shot_examples, eval_sample):
    """
    Build a prompt that includes few-shot examples and returns:
      - the combined text prompt (with examples and evaluation sample details)
      - a list of images in order: [few-shot example images..., evaluation sample image]
    The function normalizes the question text so that any image placeholder is replaced with "<image>".
    """
    prompt = "Below are examples selected from the same category as the evaluation sample. Carefully study the question and answer for each.\n\n"
    images = []
    num_examples = len(few_shot_examples)
    for idx, ex in enumerate(few_shot_examples, start=1):
        if idx < (num_examples//2 + 1):
            q_field = "image_question_0"
            a_field = "image_answer_0"
            default_answer = "First."
        else:
            q_field = "image_question_1"
            a_field = "image_answer_1"
            default_answer = "Second."

        question_text = ex[q_field].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
        
        example_text = (
            f"Example {idx}:\n"
            f"{question_text}\n"
            f"{ex.get(a_field, default_answer)}\n\n"
        )
        prompt += example_text
        images.append(ex["image_0"])
        images.append(ex["image_1"])
    
    eval_question_text_1 = eval_sample['image_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_question_text_2 = eval_sample['image_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")

    eval_prompt = "Now, based on the samples above, please answer this evaluation question. Explain your reasoning before giving the final answer:\n"
    
    prompt_1 = prompt + eval_prompt
    prompt_1 +=  f"You are given two images. {eval_question_text_1}\n"

    prompt_2 = prompt + eval_prompt
    prompt_2 +=  f"You are given two images. {eval_question_text_2}\n"


    images.append(eval_sample['image_0'])
    images.append(eval_sample['image_1'])

    return prompt_1, images, prompt_2, images

def build_prompt_without_few_shot(eval_sample):
    prompt_intro = "You are given two images. Carefully read the question and explain in one sentence your reasoning step-by-step before answering.\n\n"
    images = []
    eval_question_text_1 = eval_sample['image_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_question_text_2 = eval_sample['image_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")

    prompt_1 = prompt_intro + f"Question: {eval_question_text_1}\nAnswer: "
    prompt_2 = prompt_intro + f"Question: {eval_question_text_2}\nAnswer: "

    images.append(eval_sample['image_0'])
    images.append(eval_sample['image_1'])

    return prompt_1, images, prompt_2, images

def run_evaluation(batch_size=1, num_fewshot_examples=4, image_question=True, filename="llava_interleave_vismin_evaluation.csv", model_name='llava-hf/llava-interleave-qwen-7b-hf'):
    # load model
    llava = LlavaModel(model_name_or_path=model_name, device='cuda')

    # load dataset
    dataset = load_dataset("mair-lab/vismin-bench", split="test")
    all_samples = [s for s in dataset]

    # define task type
    if image_question:
        build_few_shot = build_few_shot_prompt_and_images_for_image_question
    else:
        build_few_shot = build_few_shot_prompt_and_images

    # some variables 
    num_samples = len(all_samples)
    num_batches = math.ceil(num_samples / batch_size)
    all_results = []

    # evaluation phase
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
        batch_samples = all_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_prompts_1 = []
        batch_image_lists_1 = []
        batch_prompts_2 = []
        batch_image_lists_2 = []
        batch_metadata = []
    
        for sample in batch_samples:
            if num_fewshot_examples:
                few_shot = get_few_shot_examples(sample, all_samples, num_examples=num_fewshot_examples)
                prompt1, images1, prompt2, images2 = build_few_shot(few_shot, sample)
            else:
                prompt1, images1, prompt2, images2 = build_prompt_without_few_shot(sample)
                
            batch_prompts_1.append(prompt1)
            batch_image_lists_1.append(images1)
            batch_prompts_2.append(prompt2)
            batch_image_lists_2.append(images2)
            batch_metadata.append({
                'sample_id': sample.get('id', None),
                'category': get_category(sample),
                'text_prompt1': prompt1,
                'text_prompt2': prompt2
            })
        
        batch_answers_1 = llava.predict_batch(texts=batch_prompts_1, image_batches=batch_image_lists_1, max_new_tokens=64)
        batch_answers_2 = llava.predict_batch(texts=batch_prompts_2, image_batches=batch_image_lists_2, max_new_tokens=64)


        for meta, ans1, ans2 in zip(batch_metadata, batch_answers_1, batch_answers_2):
            meta['answer_1'] = ans1
            meta['answer_2'] = ans2
            all_results.append(meta)
        
        print(f"Processed batch {batch_idx + 1}/{num_batches}")
        
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(filename, index=False)
    print("Evaluation complete. Results saved to " + filename)
    exit()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llava NeXT Video 7B Evaluation Pipeline")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot_examples", type=int, default=8)
    parser.add_argument("--image_question", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--filename", type=str, default="llava_interleave_vismin_evaluation.csv")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-interleave-qwen-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--load_in_Nbit", type=int, default=4)

    args = parser.parse_args()

    run_evaluation(
        batch_size=args.batch_size,
        num_fewshot_examples=args.num_fewshot_examples,
        image_question=args.image_question,
        filename=args.filename,
        model_name=args.model_name
    )
