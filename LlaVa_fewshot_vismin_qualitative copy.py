# from typing import Union
# import PIL
# import torch
# from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
# import random
# import pandas as pd
# from datasets import load_dataset
# import os
# import argparse
# from tqdm import tqdm
# from dotenv import load_dotenv
# from huggingface_hub import login

# # Load .env for API key
# load_dotenv()
# # Load from .env
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(token=HF_TOKEN)


# class LlavaModel:
#     def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_Nbit=4, **kwargs):
#         load_in_Nbit = kwargs.pop("load_in_Nbit", None)
#         if model_name_or_path == "llava-hf/llava-v1.6-mistral-7b-hf" and load_in_Nbit == 4:
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.float16,
#             )
#         else:
#             quantization_config = None
        
#         cache_dir = "./huggingface_cache"
#         self.processor = AutoProcessor.from_pretrained(
#             model_name_or_path,
#             cache_dir=cache_dir,
#             do_image_splitting=False
#         )
#         self.model = AutoModelForVision2Seq.from_pretrained(
#             model_name_or_path,
#             low_cpu_mem_usage=True,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             quantization_config=quantization_config,
#             cache_dir=cache_dir
#         )
#         self.device = device

#     def prepare_prompt(self, text, images):
#         message = [
#             {"role": "user", "content": [{"type": "text", "text": text}]}
#         ]
#         message = self.processor.apply_chat_template(message, add_generation_prompt=True)
#         return message

#     def predict(self, text: str, images: Union[PIL.Image.Image, list, str], max_new_tokens=64):
#         if isinstance(images, list):
#             if isinstance(images[0], str):
#                 images = [PIL.Image.open(img).convert("RGB") for img in images]
#         elif isinstance(images, str):
#             images = [PIL.Image.open(images).convert("RGB")]

#         prompt = self.prepare_prompt(text, images)
#         inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device, dtype=torch.float16)
#         generated_ids = self.model.generate(
#             **inputs, 
#             max_new_tokens=max_new_tokens, 
#             temperature=0.0, 
#             do_sample=False
#         )
#         if self.processor.tokenizer.padding_side == "left":
#             generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
#         output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
#         return output

#     def predict_batch(self, texts, images, max_new_tokens=20):
#         pil_images = []
#         for image in images:
#             if isinstance(image, str):
#                 image = PIL.Image.open(image).convert("RGB")
#                 pil_images.append([image])
#             elif isinstance(image, list) and isinstance(image[0], str):
#                 pil_images.append([PIL.Image.open(img).convert("RGB") for img in image])
#             else:
#                 pil_images.append(image)

#         prompts = [self.prepare_prompt(text, image) for text, image in zip(texts, images)]
#         inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(self.device, dtype=torch.float16)
#         generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
#         if self.processor.tokenizer.padding_side == "left":
#             generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
#         outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
#         return outputs

# def get_category(sample):
#     return sample.get('category', 'default')

# def get_few_shot_examples(current_sample, all_samples, num_examples=8):
#     current_cat = get_category(current_sample)
#     same_cat = [s for s in all_samples if get_category(s) == current_cat and s != current_sample]
#     return random.sample(same_cat, min(num_examples, len(same_cat)))

# def build_few_shot_prompt_and_images(few_shot_examples, eval_sample):
#     prompt = "Below are examples selected from the same category as the evaluation sample. Carefully study the question and answer for each.\n\n"
#     images = []

#     for idx, ex in enumerate(few_shot_examples, start=1):
#         if random.random() < 0.5:
#             image_field = "image_0"
#             q_field = "text_question_0"
#             a_field = "text_answer_0"
#             default_answer = "A"
#         else:
#             image_field = "image_1"
#             q_field = "text_question_1"
#             a_field = "text_answer_1"
#             default_answer = "B"

#         question_text = ex[q_field].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#         example_text = (
#             f"Example {idx}:\n"
#             f"Question: {question_text}\n"
#             f"Answer: {ex.get(a_field, default_answer)}\n\n"
#         )
#         prompt += example_text
#         images.append(ex[image_field])
    
#     eval_question_text = eval_sample['text_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#     eval_prompt = "Now, based on the patterns above, please answer this evaluation question. Explain your reasoning before giving the final answer:\n"
#     prompt_1 = prompt + eval_prompt + f"Question: {eval_question_text}\nAnswer: "

#     images_1 = images.copy()
#     images_1.append(eval_sample['image_0'])

#     eval_question_text = eval_sample['text_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#     prompt_2 = prompt + eval_prompt + f"Question: {eval_question_text}\nAnswer: "
    
#     images_2 = images.copy()
#     images_2.append(eval_sample['image_1'])

#     return prompt_1, images_1, prompt_2, images_2

# def build_few_shot_prompt_and_images_for_image_question(few_shot_examples, eval_sample):
#     """
#     Build a professional-style few-shot prompt for image-based questions.
#     """
#     prompt = "Below are examples selected from the same category as the evaluation sample. Carefully study the images and the corresponding question-answer pairs.\n\n"
#     images = []
#     num_examples = len(few_shot_examples)

#     for idx, ex in enumerate(few_shot_examples, start=1):
#         if idx < (num_examples // 2 + 1):
#             q_field = "image_question_0"
#             a_field = "image_answer_0"
#             default_answer = "First."
#         else:
#             q_field = "image_question_1"
#             a_field = "image_answer_1"
#             default_answer = "Second."

#         question_text = ex[q_field].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#         example_text = (
#             f"Example {idx}:\n"
#             f"Question: {question_text}\n"
#             f"Answer: {ex.get(a_field, default_answer)}\n\n"
#         )
#         prompt += example_text
#         images.append(ex["image_0"])
#         images.append(ex["image_1"])

#     # Now add the evaluation sample prompt
#     eval_question_text_1 = eval_sample['image_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#     eval_question_text_2 = eval_sample['image_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")

#     eval_prompt = "Now, based on the patterns above, please answer the following evaluation question. Explain your reasoning before giving the final answer:\n"

#     prompt_1 = prompt + eval_prompt + f"You are given two images. {eval_question_text_1}\nAnswer: "
#     prompt_2 = prompt + eval_prompt + f"You are given two images. {eval_question_text_2}\nAnswer: "

#     images.append(eval_sample['image_0'])
#     images.append(eval_sample['image_1'])

#     return prompt_1, images, prompt_2, images

# def build_prompt_without_few_shot(eval_sample):
#     prompt_intro = "You are given two images. Carefully read the question and explain your reasoning step-by-step before answering.\n\n"
#     images = []
#     eval_question_text_1 = eval_sample['image_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#     eval_question_text_2 = eval_sample['image_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")

#     prompt_1 = prompt_intro + f"Question: {eval_question_text_1}\nAnswer: "
#     prompt_2 = prompt_intro + f"Question: {eval_question_text_2}\nAnswer: "

#     images.append(eval_sample['image_0'])
#     images.append(eval_sample['image_1'])

#     return prompt_1, images, prompt_2, images

# def build_prompt_without_few_shot_for_text(eval_sample):
#     """
#     Build a professional zero-shot prompt for text-based caption questions.
#     """
#     prompt_intro = "You are given a caption describing a scene. Carefully read the question and explain your reasoning step-by-step before providing your final answer.\n\n"
#     images = []

#     # Prepare for first question
#     eval_question_text_1 = eval_sample['text_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#     prompt_1 = prompt_intro + f"Question: {eval_question_text_1}\nAnswer: "

#     # Prepare for second question
#     eval_question_text_2 = eval_sample['text_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
#     prompt_2 = prompt_intro + f"Question: {eval_question_text_2}\nAnswer: "

#     images.append(eval_sample['image_0'])
#     images.append(eval_sample['image_1'])

#     return prompt_1, images, prompt_2, images

# def run_evaluation(batch_size=1, num_fewshot_examples=8, image_question=True, filename="llava_vismin_evaluation_results.csv"):
#     llava = LlavaModel()
#     dataset = load_dataset("mair-lab/vismin-bench", split="test")
#     all_samples = [s for s in dataset]

#     output_dir = os.path.dirname(filename)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#     file_exists = os.path.exists(filename)

#     with open(filename, "a", encoding="utf-8") as f:
#         if not file_exists:
#             f.write("sample_id,prompt_type,prompt,generated_answer\n")

#         for eval_sample in tqdm(all_samples, desc="Evaluating samples", unit="sample"):
#             few_shot_examples = get_few_shot_examples(current_sample=eval_sample, all_samples=all_samples, num_examples=num_fewshot_examples)
            
#             # Few-shot (for few-shot settings)
#             if image_question:
#                 prompt_1_few, images_1_few, prompt_2_few, images_2_few = build_few_shot_prompt_and_images_for_image_question(few_shot_examples, eval_sample)
#             else:
#                 prompt_1_few, images_1_few, prompt_2_few, images_2_few = build_few_shot_prompt_and_images(few_shot_examples, eval_sample)

#             # Zero-shot (for zero-shot settings)
#             if image_question:
#                 prompt_1_zero, images_1_zero, prompt_2_zero, images_2_zero = build_prompt_without_few_shot(eval_sample)
#             else:
#                 prompt_1_zero, images_1_zero, prompt_2_zero, images_2_zero = build_prompt_without_few_shot_for_text(eval_sample)

#             # Few-shot prompts
#             answer_1_few = llava.predict(prompt_1_few, images_1_few)
#             answer_2_few = llava.predict(prompt_2_few, images_2_few)

#             # Zero-shot prompts
#             answer_1_zero = llava.predict(prompt_1_zero, images_1_zero)
#             answer_2_zero = llava.predict(prompt_2_zero, images_2_zero)

#             results = [
#                 {"sample_id": eval_sample["id"], "prompt_type": "few-shot", "prompt": prompt_1_few, "generated_answer": answer_1_few},
#                 {"sample_id": eval_sample["id"], "prompt_type": "few-shot", "prompt": prompt_2_few, "generated_answer": answer_2_few},
#                 {"sample_id": eval_sample["id"], "prompt_type": "zero-shot", "prompt": prompt_1_zero, "generated_answer": answer_1_zero},
#                 {"sample_id": eval_sample["id"], "prompt_type": "zero-shot", "prompt": prompt_2_zero, "generated_answer": answer_2_zero},
#             ]
#             pd.DataFrame(results).to_csv(f, mode='a', header=False, index=False)
#             f.flush()

#     print(f"Evaluation completed. Results saved to {filename}.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="LLaVA 1.6 7b Evaluation Pipeline")
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--num_fewshot_examples", type=int, default=8)
#     parser.add_argument("--image_question", type=lambda x: x.lower() == 'true', default=True)
#     parser.add_argument("--filename", type=str, default="llava_vismin_evaluation_results_captions_qualitative.csv")
#     parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--load_in_Nbit", type=int, default=4)

#     args = parser.parse_args()

#     run_evaluation(
#         batch_size=args.batch_size,
#         num_fewshot_examples=args.num_fewshot_examples,
#         image_question=args.image_question,
#         filename=args.filename
#     )



# from typing import Union
# import torch
# import PIL
# import requests
# from io import BytesIO
# import pandas as pd
# import random
# import os
# import argparse
# from tqdm import tqdm
# from dotenv import load_dotenv
# from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
# from datasets import load_dataset
# from huggingface_hub import login

# # Load .env for API key
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(token=HF_TOKEN)

# # Helper: Load image from URL
# def load_image_from_url(url):
#     response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert('RGB')

# # Llava model wrapper
# class LlavaModel:
#     def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_Nbit=4):
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#         ) if load_in_Nbit == 4 else None

#         cache_dir = "./huggingface_cache"
#         self.processor = AutoProcessor.from_pretrained(
#             model_name_or_path,
#             cache_dir=cache_dir,
#             do_image_splitting=False
#         )
#         self.model = AutoModelForVision2Seq.from_pretrained(
#             model_name_or_path,
#             low_cpu_mem_usage=True,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             quantization_config=quantization_config,
#             cache_dir=cache_dir
#         )
#         self.device = device

#     def prepare_prompt(self, prompt_text):
#         messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
#         return self.processor.apply_chat_template(messages, add_generation_prompt=True)

#     def predict_batch(self, prompts: list, images_list: list, max_new_tokens=64):
#         """Predict multiple prompts in batch."""
#         assert len(prompts) == len(images_list), "Mismatch between prompts and image lists"

#         # Prepare inputs
#         batch_inputs = self.processor(
#             text=[self.prepare_prompt(p) for p in prompts],
#             images=[
#                 [load_image_from_url(img) if isinstance(img, str) else img for img in images]
#                 for images in images_list
#             ],
#             return_tensors="pt",
#             padding=True,
#         ).to(self.device, dtype=torch.float16)

#         # Generate outputs
#         generated_ids = self.model.generate(**batch_inputs, max_new_tokens=max_new_tokens, do_sample=False)

#         if self.processor.tokenizer.padding_side == "left":
#             generated_ids = generated_ids[:, batch_inputs.input_ids.shape[1]:]

#         outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
#         return outputs

# # Build few-shot examples
# def get_few_shot_examples(eval_sample, all_samples, num_examples=8):
#     current_cat = eval_sample.get('category', 'default')
#     same_cat_samples = [s for s in all_samples if s.get('category', 'default') == current_cat and s != eval_sample]
#     return random.sample(same_cat_samples, min(num_examples, len(same_cat_samples)))

# # Build prompt depending on task
# def build_prompt(eval_sample, few_shot_examples, task="caption_to_image", few_shot=False):
#     def clean_caption_text(text):
#         return text.replace("<image_0>", "the first image").replace("<image_1>", "the second image")

#     prompt_intro = "Carefully study the following examples.\n\n" if few_shot else "You are given the following task.\n\n"
#     prompt = prompt_intro
#     images = []

#     # Few-shot examples
#     if few_shot and few_shot_examples:
#         for idx, ex in enumerate(few_shot_examples, 1):
#             if task == "caption_to_image":
#                 # ✨ Correct for caption-to-image few-shot
#                 img1 = ex['image_0']
#                 img2 = ex['image_1']
#                 question = clean_caption_text(ex['text_question_0'])
#                 answer = ex.get('text_answer_0', 'A')

#                 prompt += f"Example {idx}:\nQuestion: {question}\nImage 1: <image>\nImage 2: <image>\nAnswer: {answer}\n\n"
#                 images.extend([img1, img2])

#             elif task == "image_to_caption":
#                 # ✨ Correct for image-to-caption few-shot
#                 img = ex['image_0']
#                 caption1 = clean_caption_text(ex['image_question_0'])
#                 caption2 = clean_caption_text(ex['image_question_1'])
#                 answer = ex.get('image_answer_0', 'First.')

#                 prompt += f"Example {idx}:\nGiven an image <image>, does it best match:\n(A) {caption1}\n(B) {caption2}\nChoices: A or B.\nAnswer: {answer}\n\n"
#                 images.append(img)

#     # Evaluation sample
#     if task == "caption_to_image":
#         # ✨ Correct eval for caption-to-image
#         img1 = eval_sample['image_0']
#         img2 = eval_sample['image_1']
#         question = clean_caption_text(eval_sample['text_question_0'])

#         prompt += f"Evaluation Task:\nQuestion: {question}\nImage 1: <image>\nImage 2: <image>\nAnswer:"
#         images.extend([img1, img2])

#     elif task == "image_to_caption":
#         # ✨ Correct eval for image-to-caption
#         img = eval_sample['image_0']
#         caption1 = clean_caption_text(eval_sample['image_question_0'])
#         caption2 = clean_caption_text(eval_sample['image_question_1'])

#         prompt += f"Evaluation Task:\nGiven an image <image>, does it best match:\n(A) {caption1}\n(B) {caption2}\nChoices: A or B.\nAnswer:"
#         images.append(img)

#     # Final safety check
#     expected_image_count = prompt.count("<image>")
#     if expected_image_count != len(images):
#         raise ValueError(f"Mismatch between <image> tokens ({expected_image_count}) and provided images ({len(images)})!")

#     return prompt, images


# # Main evaluation
# def run_evaluation(task="caption_to_image", few_shot_examples=8, filename="results.csv", batch_size=8):
#     llava = LlavaModel()
#     dataset = load_dataset("mair-lab/vismin-bench", split="test")
#     all_samples = list(dataset)

#     os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
#     file_exists = os.path.exists(filename)

#     with open(filename, "a", encoding="utf-8") as f:
#         if not file_exists:
#             f.write("sample_id,prompt_type,prompt,generated_answer\n")

#         few_prompts, few_images = [], []
#         zero_prompts, zero_images = [], []
#         meta_few, meta_zero = [], []

#         for eval_sample in tqdm(all_samples, desc="Evaluating"):

#             fewshot_samples = get_few_shot_examples(eval_sample, all_samples, few_shot_examples)

#             # Few-shot
#             prompt_few, images_few = build_prompt(eval_sample, fewshot_samples, task=task, few_shot=True)
#             few_prompts.append(prompt_few)
#             few_images.append(images_few)
#             meta_few.append(eval_sample["id"])

#             # Zero-shot
#             prompt_zero, images_zero = build_prompt(eval_sample, few_shot_examples=None, task=task, few_shot=False)
#             zero_prompts.append(prompt_zero)
#             zero_images.append(images_zero)
#             meta_zero.append(eval_sample["id"])

#             # If batch full, run
#             if len(few_prompts) >= batch_size:
#                 # Few-shot
#                 few_answers = llava.predict_batch(few_prompts, few_images)
#                 # Zero-shot
#                 zero_answers = llava.predict_batch(zero_prompts, zero_images)

#                 # Write outputs
#                 rows = []
#                 for sid, p, a in zip(meta_few, few_prompts, few_answers):
#                     rows.append({"sample_id": sid, "prompt_type": "few-shot", "prompt": p, "generated_answer": a})
#                 for sid, p, a in zip(meta_zero, zero_prompts, zero_answers):
#                     rows.append({"sample_id": sid, "prompt_type": "zero-shot", "prompt": p, "generated_answer": a})

#                 pd.DataFrame(rows).to_csv(f, mode='a', header=False, index=False)
#                 f.flush()

#                 # Reset batch
#                 few_prompts, few_images = [], []
#                 zero_prompts, zero_images = [], []
#                 meta_few, meta_zero = [], []

#         # Catch leftover
#         if few_prompts:
#             few_answers = llava.predict_batch(few_prompts, few_images)
#             zero_answers = llava.predict_batch(zero_prompts, zero_images)

#             rows = []
#             for sid, p, a in zip(meta_few, few_prompts, few_answers):
#                 rows.append({"sample_id": sid, "prompt_type": "few-shot", "prompt": p, "generated_answer": a})
#             for sid, p, a in zip(meta_zero, zero_prompts, zero_answers):
#                 rows.append({"sample_id": sid, "prompt_type": "zero-shot", "prompt": p, "generated_answer": a})

#             pd.DataFrame(rows).to_csv(f, mode='a', header=False, index=False)
#             f.flush()

#     print(f"Done! Results in {filename}")
#     print(f"Evaluation completed. Results saved to {filename}.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="LLaVA Few-shot and Zero-shot Evaluation on VisMin")

#     parser.add_argument("--task", choices=["caption_to_image", "image_to_caption"], required=True)
#     parser.add_argument("--few_shot_examples", type=int, default=8)
#     parser.add_argument("--filename", type=str, default="llava_vismin_results_2.csv")
#     parser.add_argument("--batch_size", type=int, default=8)   # <<< ADD THIS LINE

#     args = parser.parse_args()

#     run_evaluation(
#         task=args.task,
#         few_shot_examples=args.few_shot_examples,
#         filename=args.filename,
#         batch_size=args.batch_size
#     )

import os
import random
import torch
import requests
import argparse
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from typing import Union
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


# === Load API Token ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)


# === Helper: Load image from URL ===
def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


# === LLaVA Model Wrapper ===
class LlavaModel:
    def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_Nbit=4):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) if load_in_Nbit == 4 else None

        cache_dir = "./huggingface_cache"
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir, do_image_splitting=False)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            cache_dir=cache_dir
        )
        self.device = device

    def prepare_prompt(self, prompt_text):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def predict_batch(self, prompts: list, images_list: list, max_new_tokens=64):
        assert len(prompts) == len(images_list)
        batch_inputs = self.processor(
            text=[self.prepare_prompt(p) for p in prompts],
            images=[
                [load_image_from_url(img) if isinstance(img, str) else img for img in images]
                for images in images_list
            ],
            return_tensors="pt",
            padding=True,
        ).to(self.device, dtype=torch.float16)

        generated_ids = self.model.generate(**batch_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, batch_inputs.input_ids.shape[1]:]

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)


# === Prompt Construction ===
def build_prompt(eval_sample, few_shot_examples, task="caption_to_image", few_shot=False):
    def clean(text): return text.replace("<image_0>", "the first image").replace("<image_1>", "the second image")

    intro = (
        "You are a vision-language assistant. Analyze the images and questions below carefully.\n\n"
    ) if few_shot else "You are given a new question. Analyze it and respond accurately.\n\n"

    prompt = intro
    images = []

    if few_shot and few_shot_examples:
        for idx, ex in enumerate(few_shot_examples, 1):
            if task == "caption_to_image":
                q = clean(ex['text_question_0'])
                a = ex.get('text_answer_0', 'first')
                prompt += f"Example {idx}:\nQuestion: {q}\nFirst image: <image>\nSecond image: <image>\nAnswer: {a}\n\n"
                images.extend([ex['image_0'], ex['image_1']])
            elif task == "image_to_caption":
                c1, c2 = clean(ex['image_question_0']), clean(ex['image_question_1'])
                a = ex.get('image_answer_0', 'first')
                prompt += f"Example {idx}:\nImage: <image>\nOption 1: {c1}\nOption 2: {c2}\nAnswer: {a}\n\n"
                images.append(ex['image_0'])

    if task == "caption_to_image":
        q = clean(eval_sample['text_question_0'])
        prompt += f"Evaluation:\nQuestion: {q}\nFirst image: <image>\nSecond image: <image>\nAnswer:"
        images.extend([eval_sample['image_0'], eval_sample['image_1']])
    elif task == "image_to_caption":
        c1 = clean(eval_sample['image_question_0'])
        c2 = clean(eval_sample['image_question_1'])
        prompt += f"Evaluation:\nImage: <image>\nOption 1: {c1}\nOption 2: {c2}\nAnswer:"
        images.append(eval_sample['image_0'])

    if prompt.count("<image>") != len(images):
        raise ValueError("Mismatch between <image> tokens and image list.")

    return prompt, images


# === Few-shot selection ===
def get_few_shot_examples(eval_sample, all_samples, k=8):
    cat = eval_sample.get('category', 'default')
    pool = [s for s in all_samples if s.get('category', 'default') == cat and s != eval_sample]
    return random.sample(pool, min(k, len(pool)))


# === Normalize answers ===
def normalize_answer(ans):
    ans = ans.lower()
    if "first" in ans or "1" in ans:
        return "first"
    if "second" in ans or "2" in ans:
        return "second"
    return "unknown"

def get_expected_answer(sample, task):
    return sample.get("text_answer_0" if task == "caption_to_image" else "image_answer_0", "first")


# === Main Evaluation + Logging ===
def run_evaluation(task="caption_to_image", few_shot_examples=8, filename="llava_results.csv", batch_size=1):
    llava = LlavaModel()
    dataset = load_dataset("mair-lab/vismin-bench", split="test")
    all_samples = list(dataset)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    file_exists = os.path.exists(filename)

    with open(filename, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("sample_id,task_type,prompt_type,generated_answer,normalized_answer,expected_answer,is_correct,prompt\n")

        batches = {"few": [], "few_img": [], "few_meta": [], "zero": [], "zero_img": [], "zero_meta": []}

        for sample in tqdm(all_samples, desc="Evaluating"):
            few_samples = get_few_shot_examples(sample, all_samples, few_shot_examples)
            p_few, imgs_few = build_prompt(sample, few_samples, task, few_shot=True)
            p_zero, imgs_zero = build_prompt(sample, None, task, few_shot=False)

            batches["few"].append(p_few)
            batches["few_img"].append(imgs_few)
            batches["few_meta"].append(sample)
            batches["zero"].append(p_zero)
            batches["zero_img"].append(imgs_zero)
            batches["zero_meta"].append(sample)

            if len(batches["few"]) >= batch_size:
                few_ans = llava.predict_batch(batches["few"], batches["few_img"])
                zero_ans = llava.predict_batch(batches["zero"], batches["zero_img"])
                all_rows = []

                for s, p, a in zip(batches["few_meta"], batches["few"], few_ans):
                    norm, truth = normalize_answer(a), get_expected_answer(s, task)
                    all_rows.append(f"{s['id']},{task},few-shot,{a},{norm},{truth},{int(norm==truth)},{p}\n")
                for s, p, a in zip(batches["zero_meta"], batches["zero"], zero_ans):
                    norm, truth = normalize_answer(a), get_expected_answer(s, task)
                    all_rows.append(f"{s['id']},{task},zero-shot,{a},{norm},{truth},{int(norm==truth)},{p}\n")

                f.writelines(all_rows)
                f.flush()
                for key in batches:
                    batches[key] = []

        print(f"Done! Results saved to {filename}.")


# === Post-evaluation analysis ===
def analyze_results(file):
    df = pd.read_csv(file)
    df["is_correct"] = df["is_correct"].astype(int)
    df["prompt_type"] = df["prompt_type"].str.lower()
    df["task_type"] = df["task_type"].str.lower()

    # Per-task accuracy
    print("\n=== Accuracy by Prompt Type and Task ===")
    print(df.groupby(["prompt_type", "task_type"])["is_correct"].mean().unstack().round(3))

    # Winoground-style group score
    results = {}
    for shot in ["few-shot", "zero-shot"]:
        d = df[df["prompt_type"] == shot].pivot(index="sample_id", columns="task_type", values="is_correct")
        d["GroupScore"] = (d["caption_to_image"] & d["image_to_caption"]).astype(int)
        results[shot] = {
            "Image Understanding": d["caption_to_image"].mean(),
            "Caption Understanding": d["image_to_caption"].mean(),
            "Group Score": d["GroupScore"].mean()
        }

    print("\n=== Winoground-Style Group Scores ===")
    print(pd.DataFrame(results).T.round(3))


# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caption_to_image", "image_to_caption"])
    parser.add_argument("--few_shot_examples", type=int, default=8)
    parser.add_argument("--filename", type=str, default="llava_results.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    run_evaluation(task=args.task, few_shot_examples=args.few_shot_examples, filename=args.filename, batch_size=args.batch_size)

    if args.analyze:
        analyze_results(args.filename)
