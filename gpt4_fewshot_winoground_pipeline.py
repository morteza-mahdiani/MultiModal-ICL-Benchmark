import os
import io
import base64
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
import requests
from openai import OpenAI
from huggingface_hub import login
import tiktoken  # NEW: for token counting

# Load .env for API key
load_dotenv()
# Load from .env
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)
class GPT4OModel:
    def __init__(self, api_key=None, model="gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)  # === COST TRACKING ===

        # === COST TRACKING ===
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_images = 0

    def _encode_images(self, images):
        encoded = []
        for img in images:
            if isinstance(img, Image.Image):
                # 🔧 Fix: convert RGBA or other non-RGB images to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")
                encoded.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_bytes}"
                    }
                })
        return encoded

    def predict(self, prompt, images=None, max_tokens=64):
        content = [{"type": "text", "text": prompt}]
        if images:
            content += self._encode_images(images)

        # === COST TRACKING ===
        input_token_count = sum(
            len(self.tokenizer.encode(part["text"])) for part in content if part["type"] == "text"
        )
        image_count = len([part for part in content if part["type"] == "image_url"])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.2
        )

        output_text = response.choices[0].message.content.strip()
        output_token_count = len(self.tokenizer.encode(output_text))

        # === COST TRACKING ===
        self.total_tokens_in += input_token_count
        self.total_tokens_out += output_token_count
        self.total_images += image_count

        return output_text

# Few-shot utilities
def get_few_shot_examples(current_sample, all_samples, num_examples=8):
    return random.sample([s for s in all_samples if s["id"] != current_sample["id"]],
                         min(num_examples, len(all_samples)-1))

def build_few_shot_prompt_and_images(few_shot_examples, eval_sample):
    prompt = "Below are examples of evaluating image-caption consistency:\n\n"
    images = []

    for i, ex in enumerate(few_shot_examples, 1):
        img = ex["image_0"]
        cap = ex["caption_0"]
        label = "Yes." if i % 2 == 0 else "No."
        prompt += f"Example {i}:\nImage: <image>\nCaption: {cap}\nAnswer: {label}\n\n"
        images.append(img)

    eval_prompts = []
    img_keys = [("image_0", "caption_0"), ("image_0", "caption_1"),
                ("image_1", "caption_0"), ("image_1", "caption_1")]

    for img_key, cap_key in img_keys:
        cap_text = eval_sample[cap_key]  # this is a string
        img_obj = eval_sample[img_key]  # this is a PIL.Image

        eval_prompt = (
                prompt +
                f"Now, evaluate this:\nImage: <image>\nCaption: {cap_text}\nAnswer: "
        )
        eval_prompts.append((eval_prompt, images + [img_obj], img_key, cap_key))

    return eval_prompts

def build_without_few_shot(eval_sample):
    prompt = "Here is an evaluation sample. Please provide the final answer in yes or no.\n\n"
    images = []

    # for i, ex in enumerate(few_shot_examples, 1):
    #     img = ex["image_0"]
    #     cap = ex["caption_0"]
    #     label = "Yes." if i % 2 == 0 else "No."
    #     prompt += f"Example {i}:\nImage: <image>\nCaption: {cap}\nAnswer: {label}\n\n"
    #     images.append(img)
    eval_prompts = []
    img_keys = [("image_0", "caption_0"), ("image_0", "caption_1"),
                ("image_1", "caption_0"), ("image_1", "caption_1")]
    for img_key, cap_key in img_keys:
        cap_text = eval_sample[cap_key]  # this is a string
        img_obj = eval_sample[img_key]  # this is a PIL.Image
        eval_prompt = (
                prompt +
                f"\nImage: <image>\nCaption: {cap_text}\nAnswer: "
        )
        eval_prompts.append((eval_prompt, images + [img_obj], img_key, cap_key))
    return eval_prompts
def download_image_from_huggingface(path):
    base_url = "https://huggingface.co/datasets/facebook/winoground/resolve/main/"
    full_url = base_url + path
    response = requests.get(full_url)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img

# Run eval
def run_evaluation(
    fewshot_k=8,
    output_path="results/gpt4o_winoground_online.csv",
    num_samples=None,
    api_key=None
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("📦 Loading Winoground from Hugging Face...")
    dataset = load_dataset("facebook/winoground", split='test')

    def load_image(example):
        for k in ["image_0", "image_1"]:
            try:
                path = example[k]  # save original image path
                print(path)
                example[k] = download_image_from_huggingface(path)
            except Exception as e:
                print(f"❌ Failed to load image at key {k} (path: {path}): {e}")
        return example

    if num_samples:
        dataset = dataset.select(range(num_samples))

    print("🖼️ Downloading and converting images...")
    # dataset = dataset.map(load_image)
    all_samples = list(dataset)
    print(all_samples[0].keys())
    model = GPT4OModel(api_key=api_key)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("sample_id,image,caption,prompt,answer\n")
        for i, eval_sample in enumerate(tqdm(all_samples, desc="Evaluating")):
            if fewshot_k:
                few_shot = get_few_shot_examples(eval_sample, all_samples, fewshot_k)
                prompt_img_pairs = build_few_shot_prompt_and_images(few_shot, eval_sample)
            else:
                prompt_img_pairs = build_without_few_shot(eval_sample)
            for idx, (prompt, images, img_key, cap_key) in enumerate(prompt_img_pairs):
                try:
                    answer = model.predict(prompt, images)
                    # answer = ''
                    image_id = "A" if "0" in img_key else "B"
                    caption_id = "1" if "0" in cap_key else "2"
                    row = pd.DataFrame([{
                        "sample_id": eval_sample["id"],
                        "image": image_id,
                        "caption": caption_id,
                        "prompt": prompt,
                        "answer": answer
                    }])
                    row.to_csv(f, header=False, index=False)
                    f.flush()
                except Exception as e:
                    print(f"❌ Failed for sample {eval_sample['id']} | {img_key}, {cap_key}: {e}")
    # === COST TRACKING ===
    input_cost = model.total_tokens_in * 0.005 / 1000
    output_cost = model.total_tokens_out * 0.015 / 1000
    image_cost = model.total_images * 0.004
    total_cost = input_cost + output_cost + image_cost

    print("\n💰 Cost Summary:")
    print(f"📥 Input tokens: {model.total_tokens_in} → ${input_cost:.4f}")
    print(f"📤 Output tokens: {model.total_tokens_out} → ${output_cost:.4f}")
    print(f"🖼️ Images sent: {model.total_images} → ${image_cost:.4f}")
    print(f"🧾 Total estimated cost: ${total_cost:.4f}")

    print(f"✅ Done! Results saved to: {output_path}")

# Run script
if __name__ == "__main__":
    run_evaluation(
        fewshot_k=8,
        output_path="results/gpt4o_winoground_online_8shot_final_RGB.csv",
        num_samples=None,  # DEBUG subset; set to None for full eval
        api_key=os.getenv("OPENAI_API_KEY")
    )
