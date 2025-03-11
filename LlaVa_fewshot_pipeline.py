from typing import Union
import PIL
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import random
import pandas as pd
import pyarrow as pa
from datasets import load_dataset, load_from_disk
import math
from tqdm import tqdm

class LlavaModel:
    def __init__(self, model_name_or_path="liuhaotian/llava-v1.6-13b", device: str = "cuda", **kwargs):
        load_in_Nbit = kwargs.pop("load_in_Nbit", None)
        
        if load_in_Nbit == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quantization_config = None
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir="../../scratch/")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            cache_dir="../../scratch/"
        )
        self.device = device
        
    def prepare_prompt(self, text, images):
        message = [{"role": "user", "content": [{"type": "text", "text": text}]}]
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2)
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
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs

# Load dataset
# dataset = load_dataset('mair-lab/vismin-bench')
# data_samples = list(dataset['test'])
# Load only the test set from local storage
dataset = load_from_disk("./vismin/hf_cache/mair-lab___vismin-bench/default/0.0.0/496293372f53df900b502f12f133fc5cad5d499a/")
print(f"Loaded {len(dataset)} test samples")
data_samples = list(dataset['test'])


# dataset = load_dataset("mair-lab/vismin-bench", cache_dir="./vismin/hf_cache/", local_files_only=True)
# dataset = load_from_disk("./vismin-bench")
# print(f"Loaded {len(dataset)} test samples")
# data_samples = list(dataset['test'])

# Benchmarking configuration
batch_size = 6
num_samples = len(data_samples)
num_batches = math.ceil(num_samples / batch_size)

llava = LlavaModel(device="cuda")
all_results = []

for batch_idx in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
    batch_samples = data_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    batch_prompts = []
    batch_images = []
    batch_metadata = []
 
    for sample in batch_samples:
        prompt = sample['text_question_0'].replace("<image_0>", "<image>")
        batch_prompts.append(prompt)
        batch_images.append([sample['image_0']])
        batch_metadata.append({
            'sample_id': sample.get('id', None),
            'category': sample.get('category', 'default'),
            'text_prompt': prompt  
        })
    
    batch_answers = llava.predict_batch(texts=batch_prompts, images=batch_images, max_new_tokens=64)
    
    for meta, ans in zip(batch_metadata, batch_answers):
        meta['text_answer'] = ans
        all_results.append(meta)
    
    print(f"Processed batch {batch_idx + 1}/{num_batches}")

results_df = pd.DataFrame(all_results)
results_df.to_csv('llava1_6_vismin_evaluation_results.csv', index=False)
print("Evaluation complete. Results saved to llava1_6_vismin_evaluation_results.csv")
