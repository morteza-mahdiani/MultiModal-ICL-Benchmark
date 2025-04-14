# import os
# import torch
# import pandas as pd
# from tqdm import tqdm
# from datasets import load_dataset
# from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
# from PIL import Image
# import random
# from dotenv import load_dotenv

# # Load Hugging Face token from .env file
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# class LlavaModel:
#     def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_Nbit=4):
#         quant_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#         ) if load_in_Nbit == 4 else None

#         self.device = device
#         self.processor = AutoProcessor.from_pretrained(model_name_or_path)
#         self.model = AutoModelForVision2Seq.from_pretrained(
#             model_name_or_path,
#             quantization_config=quant_config,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )

#     def prepare_prompt(self, caption):
#         prompt = (
#             "You are shown an image. Please answer whether the caption below correctly describes the image."
#             f"Caption: {caption} Answer:"
#         )
#         return prompt

#     def predict(self, text_prompt, images, max_new_tokens=64):
#         inputs = self.processor(
#             text=text_prompt,
#             images=images[-1],
#             return_tensors="pt"
#         ).to(self.device, torch.float16)
#         generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
#         if self.processor.tokenizer.padding_side == "left":
#             generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
#         return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()


# def get_few_shot_examples(current_sample, all_samples, num_examples=8):
#     candidates = [s for s in all_samples if s != current_sample]
#     return random.sample(candidates, min(num_examples, len(candidates)))


# def build_few_shot_prompt_and_images_winoground(few_shot_examples, eval_sample):
#     prompt = "Below are examples where the model determines if a caption correctly describes an image:\n\n"
#     images = []

#     for idx, ex in enumerate(few_shot_examples, start=1):
#         image = ex["image_0"]
#         caption = ex["caption_0"]
#         answer = "Yes." if idx % 2 == 0 else "No."
#         prompt += f"Example {idx}:\nImage: <image>\nCaption: {caption}\nAnswer: {answer}\n\n"
#         images.append(image)

#     prompts_images = []
#     for img_key, cap_key in [("image_0", "caption_0"), ("image_0", "caption_1"),
#                              ("image_1", "caption_0"), ("image_1", "caption_1")]:
#         cap = eval_sample[cap_key]
#         img = eval_sample[img_key]
#         eval_prompt = (
#             prompt +
#             f"Now, evaluate the following:\nImage: <image>\nCaption: {cap}\nAnswer: "
#         )
#         all_images = images + [img]
#         prompts_images.append((eval_prompt, all_images, img_key, cap_key))

#     return prompts_images


# from datetime import datetime

# def run_evaluation(output_path=None, batch_size=1, fewshot_k=8):
#     model_name = "llava_v1.6"
#     if output_path is None:
#         output_path = f"results/{model_name}_winoground.csv"
#     model = LlavaModel()

#     if HF_TOKEN is None:
#         raise ValueError("HF_TOKEN not found. Make sure .env file is loaded correctly.")

#     dataset = load_dataset(
#         "json",
#         data_files={"test": "data/examples.jsonl"},
#         split="test"
#     )

#     # Load and unzip images manually if not done already
#     # Make sure all images are stored in data/images/ with correct filenames matching the JSON entries

#     def load_image(example):
#         for key in ["image_0", "image_1"]:
#             image_path = os.path.join("data/images", example[key])
#             image_path += ".png"
#             try:
#                 example[key] = Image.open(image_path).convert("RGB")
#             except Exception as e:
#                 print(f"❌ Failed to load {image_path}: {e}")
#                 raise
#         return example

#     dataset = dataset.select(range(5))  # Limit to 5 test samples for debugging
#     dataset = dataset.map(
#         load_image,
#         load_from_cache_file=False,
#         num_proc=1,
#         desc="Loading and validating images"
#     )
#     print("✅ Yeeeees — All images loaded successfully")
#     all_samples = list(dataset)

#     completed_ids = set()
#     if os.path.exists(output_path) and os.stat(output_path).st_size > 0:
#         completed_df = pd.read_csv(output_path)
#         completed_ids = set(completed_df["sample_id"].unique())
#         print(f"Resuming from previous run. {len(completed_ids)} samples already completed.")

#     if os.path.dirname(output_path):
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "a" if os.path.exists(output_path) else "w", encoding="utf-8") as f:
#         if os.stat(output_path).st_size == 0:
#             f.write("sample_id,image,caption,prompt,answer\n")

#         for eval_idx, eval_sample in enumerate(tqdm(all_samples, desc="Evaluating Winoground", unit="sample")):
#             if eval_sample["id"] in completed_ids:
#                 continue

#             few_shot_examples = get_few_shot_examples(eval_sample, all_samples, num_examples=fewshot_k)
#             print(f"\n--- Sample {eval_idx+1}/{len(all_samples)} | Sample ID: {eval_sample['id']} ---")
#             print(f"Using {len(few_shot_examples)} few-shot examples.")

#             prompts_images = build_few_shot_prompt_and_images_winoground(few_shot_examples, eval_sample)

#             for pair_idx, (prompt, images, img_key, cap_key) in enumerate(prompts_images):
#                 # try:
#                     print(f"  ▶ Pair {pair_idx+1}/4 → Image: {img_key}, Caption: {cap_key}")
#                     answer = model.predict(prompt, images)
#                     print(answer)
#                 #     image_idx = "A" if img_key == "image_0" else "B"
#                 #     caption_idx = "1" if cap_key == "caption_0" else "2"
#                 #     row = pd.DataFrame([{
#                 #         "sample_id": eval_sample["id"],
#                 #         "image": image_idx,
#                 #         "caption": caption_idx,
#                 #         "prompt": prompt,
#                 #         "answer": answer
#                 #     }])
#                 #     row.to_csv(f, header=False, index=False)
#                 #     f.flush()
#                 # except Exception as e:
#                 #     print(f"❌ Error on sample {eval_sample['id']} - {img_key}, {cap_key}: {e}")
#                 #     continue


# if __name__ == "__main__":
#     run_evaluation()

import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import random
from dotenv import load_dotenv

# Load Hugging Face token from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class LlavaModel:
    def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_Nbit=4):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) if load_in_Nbit == 4 else None

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def prepare_prompt(self, caption):
        prompt = (
            "You are shown an image. Please answer whether the caption below correctly describes the image."
            f"Caption: {caption} Answer:"
        )
        return prompt

    def predict(self, text_prompt, images, max_new_tokens=64):
        inputs = self.processor(
            images=images[-1],
            messages=[{"role": "user", "content": text_prompt}],
            return_tensors="pt"
        ).to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def get_few_shot_examples(current_sample, all_samples, num_examples=8):
    candidates = [s for s in all_samples if s != current_sample]
    return random.sample(candidates, min(num_examples, len(candidates)))


def build_few_shot_prompt_and_images_winoground(few_shot_examples, eval_sample):
    prompt = "Below are examples where the model determines if a caption correctly describes an image:\n\n"
    images = []

    for idx, ex in enumerate(few_shot_examples, start=1):
        image = ex["image_0"]
        caption = ex["caption_0"]
        answer = "Yes." if idx % 2 == 0 else "No."
        prompt += f"Example {idx}:\nImage: <image>\nCaption: {caption}\nAnswer: {answer}\n\n"
        images.append(image)

    prompts_images = []
    for img_key, cap_key in [("image_0", "caption_0"), ("image_0", "caption_1"),
                             ("image_1", "caption_0"), ("image_1", "caption_1")]:
        cap = eval_sample[cap_key]
        img = eval_sample[img_key]
        eval_prompt = (
            prompt +
            f"Now, evaluate the following:\nImage: <image>\nCaption: {cap}\nAnswer: "
        )
        all_images = images + [img]
        prompts_images.append((eval_prompt, all_images, img_key, cap_key))

    return prompts_images


from datetime import datetime

def run_evaluation(output_path=None, batch_size=1, fewshot_k=8):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "llava_v1.6"
    if output_path is None:
        output_path = f"results/{model_name}_winoground_{timestamp}.csv"
    model = LlavaModel()

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not found. Make sure .env file is loaded correctly.")

    dataset = load_dataset(
        "json",
        data_files={"test": "data/examples.jsonl"},
        split="test"
    )

    # Load and unzip images manually if not done already
    # Make sure all images are stored in data/images/ with correct filenames matching the JSON entries

    def load_image(example):
        for key in ["image_0", "image_1"]:
            image_path = os.path.join("data/images", example[key])
            image_path += ".png"
            try:
                example[key] = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"❌ Failed to load {image_path}: {e}")
                raise
        return example

    dataset = dataset.select(range(5))  # Limit to 5 test samples for debugging
    dataset = dataset.map(
        load_image,
        load_from_cache_file=False,
        num_proc=1,
        desc="Loading and validating images"
    )
    print("✅ Yeeeees — All images loaded successfully")
    all_samples = list(dataset)

    completed_ids = set()
    if os.path.exists(output_path) and os.stat(output_path).st_size > 0:
        completed_df = pd.read_csv(output_path)
        completed_ids = set(completed_df["sample_id"].unique())
        print(f"Resuming from previous run. {len(completed_ids)} samples already completed.")

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a" if os.path.exists(output_path) else "w", encoding="utf-8") as f:
        if os.stat(output_path).st_size == 0:
            f.write("sample_id,image,caption,prompt,answer\n")

        for eval_idx, eval_sample in enumerate(tqdm(all_samples, desc="Evaluating Winoground", unit="sample")):
            if eval_sample["id"] in completed_ids:
                continue

            few_shot_examples = get_few_shot_examples(eval_sample, all_samples, num_examples=fewshot_k)
            print(f"\n--- Sample {eval_idx+1}/{len(all_samples)} | Sample ID: {eval_sample['id']} ---")
            print(f"Using {len(few_shot_examples)} few-shot examples.")

            prompts_images = build_few_shot_prompt_and_images_winoground(few_shot_examples, eval_sample)

            for pair_idx, (prompt, images, img_key, cap_key) in enumerate(prompts_images):
                try:
                    print(f"  ▶ Pair {pair_idx+1}/4 → Image: {img_key}, Caption: {cap_key}")
                    answer = model.predict(prompt, images)
                    image_idx = "A" if img_key == "image_0" else "B"
                    caption_idx = "1" if cap_key == "caption_0" else "2"
                    row = pd.DataFrame([{
                        "sample_id": eval_sample["id"],
                        "image": image_idx,
                        "caption": caption_idx,
                        "prompt": prompt,
                        "answer": answer
                    }])
                    row.to_csv(f, header=False, index=False)
                    f.flush()
                except Exception as e:
                    print(f"❌ Error on sample {eval_sample['id']} - {img_key}, {cap_key}: {e}")
                    continue


if __name__ == "__main__":
    run_evaluation()
