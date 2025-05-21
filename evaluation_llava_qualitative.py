# Re-import everything and re-run the analysis
import pandas as pd

# Load the file again
file_path = '/mnt/data/llava_vismin_evaluation_results_captions_qualitative.csv'
df = pd.read_csv(file_path)

# Sort by sample_id and prompt_type for consistency
df_sorted = df.sort_values(by=['sample_id', 'prompt_type']).reset_index(drop=True)

# Generate expected labels: for each sample_id (4 rows expected: few-shot A, few-shot B, zero-shot A, zero-shot B)
expected_labels = []
for idx in range(0, len(df_sorted), 4):
    expected_labels.extend([
        "A",  # few-shot first (image A)
        "B",  # few-shot second (image B)
        "A",  # zero-shot first (image A)
        "B"   # zero-shot second (image B)
    ])

# Add expected label to dataframe
df_sorted['expected_label'] = expected_labels

# Define function to extract predicted label from the generated answer
def extract_predicted_label(text):
    text = text.lower()
    if "(a)" in text or "option a" in text or "first" in text:
        return "A"
    if "(b)" in text or "option b" in text or "second" in text:
        return "B"
    return "Unknown"

# Apply the function to get predicted labels
df_sorted['predicted_label'] = df_sorted['generated_answer'].apply(extract_predicted_label)

# Compare predicted vs expected
df_sorted['correct'] = df_sorted['predicted_label'] == df_sorted['expected_label']

# Now calculate separate accuracies for few-shot and zero-shot
few_shot_accuracy = df_sorted[df_sorted['prompt_type'] == 'few-shot']['correct'].mean()
zero_shot_accuracy = df_sorted[df_sorted['prompt_type'] == 'zero-shot']['correct'].mean()

few_shot_accuracy, zero_shot_accuracy
