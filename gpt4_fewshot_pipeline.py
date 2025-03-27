from openai import OpenAI
import random
import pandas as pd
from datasets import load_dataset
import math
from tqdm import tqdm
import argparse
import base64
from PIL import Image
import io

# --- GPT-4-O model wrapper ---
class GPT4OModel:
    def __init__(self, client, model="gpt-4o", api_key=None):
        # Assumes the client is initialized with the API key
        self.client = client
        self.model = model

    def predict(self, prompt: str, images=None, max_tokens=64):
        """
        Predict using GPT-4-O by sending both a text prompt and associated image(s).
        The 'images' parameter should be a list of image file paths or PIL Image objects.
        """
        if images is None:
            images = []
        
        # Process images if they are file paths
        processed_images = []
        for img in images:
            if isinstance(img, str):
                # If it's a file path, open and process the image
                with open(img, 'rb') as image_file:
                    processed_images.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                        }
                    })
            elif isinstance(img, Image.Image):
                # If it's a PIL Image, convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                processed_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                    }
                })

        # Create the message with both text and images
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + processed_images
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[message],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def predict_batch(self, prompts, images_list, max_tokens=64):
        """
        Batch prediction by looping over prompts and associated images.
        """
        outputs = []
        for prompt, images in zip(prompts, images_list):
            outputs.append(self.predict(prompt, images, max_tokens))
        return outputs


# --- Few-shot example and prompt construction functions ---
def get_category(sample):
    return sample.get('category', 'default')


def get_few_shot_examples(current_sample, all_samples, num_examples=1):
    current_cat = get_category(current_sample)
    same_cat = [s for s in all_samples if get_category(s) == current_cat and s != current_sample]
    return random.sample(same_cat, min(num_examples, len(same_cat)))


def build_few_shot_prompt(few_shot_examples, eval_sample):
    """
    Build a prompt that includes few-shot examples and returns:
      - prompt_1: text prompt (version 1)
      - images_1: evaluation image(s) corresponding to version 1
      - prompt_2: text prompt (version 2)
      - images_2: evaluation image(s) corresponding to version 2
    For GPT-4-O, we attach the evaluation images alongside the text.
    """
    prompt = "Below are some examples demonstrating how to answer based on the provided captions and questions:\n\n"

    for idx, ex in enumerate(few_shot_examples, start=1):
        # Choose fields randomly to simulate two possible formats.
        if random.random() < 0.5:
            q_field = "text_question_0"
            a_field = "text_answer_0"
            default_answer = "A"
        else:
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

    # Build evaluation prompt and associate the corresponding images.
    eval_question_text = eval_sample['text_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_prompt = "Now, please answer the following for the evaluation sample:\n"
    eval_prompt += f"Question: {eval_question_text}\n"
    eval_prompt += "Answer: "

    prompt_1 = prompt + eval_prompt
    # For version 1, attach image_0 from the evaluation sample.
    images_1 = [eval_sample['image_0']]

    # For demonstration, create a second variant using the alternate image.
    eval_question_text2 = eval_sample['text_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    eval_prompt2 = "Now, please answer the following for the evaluation sample:\n"
    eval_prompt2 += f"Question: {eval_question_text2}\n"
    eval_prompt2 += "Answer: "
    prompt_2 = prompt + eval_prompt2
    images_2 = [eval_sample['image_1']]

    return prompt_1, images_1, prompt_2, images_2


def build_prompt_without_few_shot(eval_sample):
    """
    Build a prompt without few-shot examples and return two variants with their corresponding images.
    """
    prompt = "You are given an image and a question. "
    eval_question_text = eval_sample['text_question_0'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    prompt += f"Question: {eval_question_text}\nAnswer: "
    images_1 = [eval_sample['image_0']]

    prompt_2 = "You are given an image and a question. "
    eval_question_text2 = eval_sample['text_question_1'].replace("<image_0>", "<image>").replace("<image_1>", "<image>")
    prompt_2 += f"Question: {eval_question_text2}\nAnswer: "
    images_2 = [eval_sample['image_1']]

    return prompt, images_1, prompt_2, images_2


# --- Evaluation pipeline ---
def run_evaluation(
    finetuned=False, 
    batch_size=2, 
    num_fewshot_examples=1, 
    use_image_question=True, 
    filename="gpt4o_vismin_evaluation_results.csv",
    subset_size=None,
    api_key=None
):
    client = OpenAI(api_key=api_key)
    # Instantiate the GPT-4-O model wrapper.
    model = GPT4OModel(client, api_key=api_key)

    # Select the prompt builder function.
    if use_image_question:
        build_prompt_fn = build_few_shot_prompt
    else:
        build_prompt_fn = build_prompt_without_few_shot

    # Load the dataset.
    dataset = load_dataset('mair-lab/vismin-bench')
    data_samples = list(dataset['test'])
    if subset_size is not None:
        data_samples = data_samples[:subset_size]
    num_samples = len(data_samples)
    num_batches = math.ceil(num_samples / batch_size)

    all_results = []
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
        batch_prompts_1 = []
        batch_images_1 = []
        batch_prompts_2 = []
        batch_images_2 = []
        batch_metadata = []

        batch_samples = data_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        for sample in batch_samples:
            if num_fewshot_examples:
                few_shot = get_few_shot_examples(sample, data_samples, num_examples=num_fewshot_examples)
                prompt1, images1, prompt2, images2 = build_prompt_fn(few_shot, sample)
            else:
                prompt1, images1, prompt2, images2 = build_prompt_without_few_shot(sample)
            batch_prompts_1.append(prompt1)
            batch_images_1.append(images1)
            batch_prompts_2.append(prompt2)
            batch_images_2.append(images2)
            batch_metadata.append({
                'sample_id': sample.get('id', None),
                'category': get_category(sample),
                'text_prompt1': prompt1,
                'text_prompt2': prompt2
            })

            print('@@@@@@@@@@@@@@@@@@@@@@@')
            print("Prompt 1:")
            print(prompt1)
            print("Images 1:", images1)
            print('@@@@@@@@@@@@@@@@@@@@@@@')
            print("Prompt 2:")
            print(prompt2)
            print("Images 2:", images2)
            print('@@@@@@@@@@@@@@@@@@@@@@@')

        # Use GPT-4-O for predictions.
        batch_answers_1 = model.predict_batch(prompts=batch_prompts_1, images_list=batch_images_1, max_tokens=64)
        batch_answers_2 = model.predict_batch(prompts=batch_prompts_2, images_list=batch_images_2, max_tokens=64)

        for meta, ans1, ans2 in zip(batch_metadata, batch_answers_1, batch_answers_2):
            meta['text_answer_1'] = ans1
            meta['text_answer_2'] = ans2
            all_results.append(meta)

        print(f"Processed batch {batch_idx + 1}/{num_batches}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(filename, index=False)
    print("Evaluation complete. Results saved to " + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT-4-O Evaluation Pipeline")

    parser.add_argument("--finetuned", type=lambda x: x.lower() == 'true', default=False, help="Flag for finetuned model (ignored for GPT-4-O)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_fewshot_examples", type=int, default=8, help="Number of few-shot examples to use")
    parser.add_argument("--use_image_question", type=lambda x: x.lower() == 'true', default=True, help="Set True for image-based questions, False for text-based")
    parser.add_argument("--filename", type=str, default="gpt4o_vismin_evaluation_results.csv", help="Output filename")
    parser.add_argument("--subset_size", type=int, default=10, help="Limit evaluation to a subset of the dataset")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key for GPT-4-O access")

    args = parser.parse_args()

    run_evaluation(
        finetuned=args.finetuned,
        batch_size=args.batch_size,
        num_fewshot_examples=args.num_fewshot_examples,
        use_image_question=args.use_image_question,
        filename=args.filename,
        subset_size=args.subset_size,
        api_key=args.api_key
    )
