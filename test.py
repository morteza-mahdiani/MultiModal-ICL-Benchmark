from datasets import load_from_disk

# dataset_path = "/Users/mortezamahdiani/Downloads/mair_vismin/vismin_bench/data/"
# dataset = load_from_disk(dataset_path)

# print(dataset)

from datasets import load_dataset
from datasets import load_dataset

# Download and save dataset
dataset = load_dataset("mair-lab/vismin-bench")
dataset.save_to_disk("vismin_bench")  # This ensures all files are included

# dataset = load_dataset("mair-lab/vismin-bench", cache_dir="./hf_cache")

# dataset = load_dataset("mair-lab/vismin-bench", cache_dir="./hf_cache")
# print(len(dataset['test']))
# print(dataset)