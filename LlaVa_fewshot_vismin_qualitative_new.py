from typing import Union
import PIL
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import random
import pandas as pd
from datasets import load_dataset
import os
import argparse
from tqdm import tqdm

class LlavaModel:
    def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_Nbit=4, **kwargs):
        load_in_Nbit = kwargs.pop("load_in_Nbit", None)
        if model_name_or_path == "llava-hf/llava-v1.6-mistral-7b-hf" and load_in_Nbit == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quantization_config = None
        
        cache_dir = "./huggingface_cache"
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            do_image_splitting=False
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            cache_dir=cache_dir
        )
        self.device = device

    def prepare_prompt(self, text, images):
        message = [
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        message = self.processor.apply_chat_template(message, add_generation_prompt=True)
        return message

    def predict(self, text: str, images: Union[PIL.Image.Image, list, str], max_new_tokens=64):
        if isinstance(images, list):
            if isinstance(images[0], str):
                images = [PIL.Image.open(img).convert("RGB") for img in images]
        elif isinstance(images, str):
            images = [PIL.Image.open(images).convert("RGB")]

        prompt = self.prepare_prompt(text, images)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.0, 
            do_sample=False
        )
        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return output

    def predict_batch(self, texts, images, max_new_tokens=20):
        pil_images = []
        for image in images:
            if isinstance(image, str):
                image = PIL.Image.open(image).convert("RGB")
                pil_images.append([image])
            elif isinstance(image, list) and isinstance(image[0], str):
                pil_images.append([PIL.Image.open(img).convert("RGB") for img in image])
            else:
                pil_images.append(image)

        prompts = [self.prepare_prompt(text, image) for text, image in zip(texts, images)]
        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
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
    eval_prompt = "Now, based on the patterns above, please answer this evaluation question. Explain your reasoning before giving the final answer:\n"
    prompt_1 = prompt + eval_prompt + f"Question: {eval_question_text}\nAnswer: "

    images_1 = images.copy()
    images_1.append(eval_sample['image_0'])

    eval_question_text = eval_sample['text_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    prompt_2 = prompt + eval_prompt + f"Question: {eval_question_text}\nAnswer: "
    
    images_2 = images.copy()
    images_2.append(eval_sample['image_1'])

    return prompt_1, images_1, prompt_2, images_2

def build_prompt_without_few_shot(eval_sample):
    prompt_intro = "You are given two images. Carefully read the question and explain your reasoning step-by-step before answering.\n\n"
    images = []
    eval_question_text_1 = eval_sample['image_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_question_text_2 = eval_sample['image_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")

    prompt_1 = prompt_intro + f"Question: {eval_question_text_1}\nAnswer: "
    prompt_2 = prompt_intro + f"Question: {eval_question_text_2}\nAnswer: "

    images.append(eval_sample['image_0'])
    images.append(eval_sample['image_1'])

    return prompt_1, images, prompt_2, images

def run_evaluation(batch_size=1, num_fewshot_examples=8, image_question=True, filename="llava_vismin_evaluation_results.csv"):
    llava = LlavaModel()
    dataset = load_dataset("mair-lab/vismin-bench", split="test")
    all_samples = [s for s in dataset]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.exists(filename)

    with open(filename, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("sample_id,prompt_type,prompt,generated_answer\n")

        for eval_sample in tqdm(all_samples, desc="Evaluating samples", unit="sample"):
            few_shot_examples = get_few_shot_examples(current_sample=eval_sample, all_samples=all_samples, num_examples=num_fewshot_examples)
            
            # Few-shot prompts
            if image_question:
                prompt_1_few, images_1_few, prompt_2_few, images_2_few = build_few_shot_prompt_and_images(few_shot_examples, eval_sample)
            else:
                prompt_1_few, images_1_few, prompt_2_few, images_2_few = build_few_shot_prompt_and_images(few_shot_examples, eval_sample)

            answer_1_few = llava.predict(prompt_1_few, images_1_few)
            answer_2_few = llava.predict(prompt_2_few, images_2_few)

            # Zero-shot prompts
            prompt_1_zero, images_1_zero, prompt_2_zero, images_2_zero = build_prompt_without_few_shot(eval_sample)
            answer_1_zero = llava.predict(prompt_1_zero, images_1_zero)
            answer_2_zero = llava.predict(prompt_2_zero, images_2_zero)

            results = [
                {"sample_id": eval_sample["id"], "prompt_type": "few-shot", "prompt": prompt_1_few, "generated_answer": answer_1_few},
                {"sample_id": eval_sample["id"], "prompt_type": "few-shot", "prompt": prompt_2_few, "generated_answer": answer_2_few},
                {"sample_id": eval_sample["id"], "prompt_type": "zero-shot", "prompt": prompt_1_zero, "generated_answer": answer_1_zero},
                {"sample_id": eval_sample["id"], "prompt_type": "zero-shot", "prompt": prompt_2_zero, "generated_answer": answer_2_zero},
            ]
            pd.DataFrame(results).to_csv(f, mode='a', header=False, index=False)
            f.flush()

    print(f"Evaluation completed. Results saved to {filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA 1.6 7b Evaluation Pipeline")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot_examples", type=int, default=8)
    parser.add_argument("--image_question", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--filename", type=str, default="llava_vismin_evaluation_results_captions.csv")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_in_Nbit", type=int, default=4)

    args = parser.parse_args()

    run_evaluation(
        batch_size=args.batch_size,
        num_fewshot_examples=args.num_fewshot_examples,
        image_question=args.image_question,
        filename=args.filename
    )
