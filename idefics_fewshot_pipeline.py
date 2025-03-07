from typing import Union
import PIL
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import random
import pandas as pd
from datasets import load_dataset
import math
from tqdm import tqdm
import argparse

class Idefics2Model:
    def __init__(self, model_name_or_path="HuggingFaceM4/idefics2-8b", device: str = "cuda", **kwargs):
        attn_implementation = None
        load_in_Nbit = kwargs.pop("load_in_Nbit", None)
        if model_name_or_path == "HuggingFaceM4/idefics2-8b" and load_in_Nbit == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quantization_config = None
        
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", cache_dir="../../scratch/", do_image_splitting=False)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            _attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            cache_dir="../../scratch/"
        )
        self.device = device
        
        
    def prepare_prompt(self, text, images):
        """
        Prepares the prompt for few-shot evaluation.
        Here, `text` is assumed to be the complete prompt (including few-shot examples)
        and `images` is a list of PIL.Images. The order of images should match the order of
        few-shot examples followed by the evaluation sample image.
        """
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ],
            }
        ]
        
        message = self.processor.apply_chat_template(message, add_generation_prompt=True)
        return message

    def predict(self, text: str, images: Union[PIL.Image.Image, list, str], max_new_tokens=64):
        """
        Predict using a fully built few-shot prompt (text) and the corresponding images.
        The images argument can be a list of file paths or PIL.Image objects.
        """
        
        if isinstance(images, list):
            if isinstance(images[0], str):
                images = [PIL.Image.open(img).convert("RGB") for img in images]
        elif isinstance(images, str):
            images = [PIL.Image.open(images).convert("RGB")]
        
        prompt = self.prepare_prompt(text, images)
        print(prompt)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        print(generated_ids[0])
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return output

    def predict_batch(self, texts, images, max_new_tokens=20):
        """
        Batch prediction for multiple prompts.
        """
        if isinstance(images, list):
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
        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(
            self.device, dtype=torch.float16
        )
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
    """
    Build a prompt that includes few-shot examples and returns:
      - the combined text prompt (with examples and evaluation sample details)
      - a list of images in order: [few-shot example images..., evaluation sample image]
    The function normalizes the question text so that any image placeholder is replaced with "<image>".
    """
    prompt = "Below are some examples demonstrating how to answer based on the provided captions and images:\n\n"
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
    
    eval_prompt = "Now, please answer the following for the evaluation sample:\n"
    eval_prompt += f"Question: {eval_question_text}\n"
    eval_prompt += "Answer: "
    
    prompt_1 = prompt + eval_prompt
    images_1 = images.copy()  
    images_1.append(eval_sample['image_0'])

    prompt_2 = prompt + eval_prompt
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
    prompt = "Below are some examples demonstrating how to answer based on the provided images and question:\n\n"
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

    eval_prompt = "Now, please answer the following for the evaluation sample:\n"
    
    prompt_1 = prompt + eval_prompt
    prompt_1 +=  f"You are given two images. {eval_question_text_1}\n"

    prompt_2 = prompt + eval_prompt
    prompt_2 +=  f"You are given two images. {eval_question_text_2}\n"


    images.append(eval_sample['image_0'])
    images.append(eval_sample['image_1'])

    return prompt_1, images, prompt_2, images

def build_prompt_without_few_shot(eval_sample):
    prompt = "You are given two images."
    images = []
    eval_question_text_1 = eval_sample['image_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_question_text_2 = eval_sample['image_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    
    prompt_1 = prompt + f"{eval_question_text_1}\n"
    prompt_1 += "Answer: "

    prompt_2 = prompt + f"{eval_question_text_2}\n"
    prompt_2 += "Answer: "

    images.append(eval_sample['image_0'])
    images.append(eval_sample['image_1'])

    return prompt_1, images, prompt_2, images

def run_evaluation(batch_size=16, num_fewshot_examples=8, image_question=True, filename="idefics2_vismin_evaluation_results.csv"):
    idefics2 = Idefics2Model(device="cuda")
    if image_question:
        build_few_shot = build_few_shot_prompt_and_images_for_image_question
    else:
        build_few_shot = build_few_shot_prompt_and_images
    
    dataset = load_dataset('mair-lab/vismin-bench')
    data_samples = list(dataset['test'])
    num_samples = len(data_samples)
    num_batches = math.ceil(num_samples / batch_size)
    all_results = []
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
        batch_samples = data_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_prompts_1 = []
        batch_image_lists_1 = []
        batch_prompts_2 = []
        batch_image_lists_2 = []
        batch_metadata = []
    
        for sample in batch_samples:
            if num_fewshot_examples:
                few_shot = get_few_shot_examples(sample, data_samples, num_examples=num_fewshot_examples)
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
        
        batch_answers_1 = idefics2.predict_batch(texts=batch_prompts_1, images=batch_image_lists_1, max_new_tokens=64)
        batch_answers_2 = idefics2.predict_batch(texts=batch_prompts_2, images=batch_image_lists_2, max_new_tokens=64)

        for meta, ans1, ans2 in zip(batch_metadata, batch_answers_1, batch_answers_2):
            meta['text_answer_1'] = ans1
            meta['text_answer_2'] = ans2
            all_results.append(meta)
        
        print(f"Processed batch {batch_idx + 1}/{num_batches}")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(filename, index=False)
    print("Evaluation complete. Results saved to " + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Idefics2 pipeline")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_fewshot_examples", type=int, default=8, help="Number of few-shot examples")
    parser.add_argument("--image_question", type=lambda x: x.lower() == 'true', default=True, help="Set True for image-based questions, False for text-based")
    parser.add_argument("--filename", type=str, default="idefics2_vismin_evaluation_results.csv", help="Output filename")
    
    args = parser.parse_args()
    
    run_evaluation(
        batch_size=args.batch_size,
        num_fewshot_examples=args.num_fewshot_examples,
        image_question=args.image_question,
        filename=args.filename
    )

