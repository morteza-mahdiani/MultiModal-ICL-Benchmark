import pandas as pd

def evaluate_winoground(results_path):
    df = pd.read_csv(results_path)

    # Normalize answers
    df['answer'] = df['answer'].str.lower().str.strip()
    df['is_yes'] = df['answer'].str.startswith("yes")

    # Pivot to get all 4 conditions per sample
    grouped = df.groupby('sample_id')

    total = 0
    correct = 0
    detailed_results = []

    for sample_id, group in grouped:
        if len(group) != 4:
            print(f"⚠️ Skipping sample {sample_id} due to incomplete pairs")
            continue

        yes_flags = {
            (row['image'], row['caption']): row['is_yes']
            for _, row in group.iterrows()
        }

        cond_1 = yes_flags.get(('A', '1'), False)  # image_0 & caption_0 → Yes
        cond_2 = yes_flags.get(('B', '2'), False)  # image_1 & caption_1 → Yes
        cond_3 = not yes_flags.get(('A', '2'), True)  # image_0 & caption_1 → No
        cond_4 = not yes_flags.get(('B', '1'), True)  # image_1 & caption_0 → No
        print(cond_1, cond_2, cond_3, cond_4)
        is_correct = all([cond_1, cond_2, cond_3, cond_4])

        detailed_results.append({
            "sample_id": sample_id,
            "image0_caption0": yes_flags.get(('A', '1'), None),
            "image0_caption1": yes_flags.get(('A', '2'), None),
            "image1_caption0": yes_flags.get(('B', '1'), None),
            "image1_caption1": yes_flags.get(('B', '2'), None),
            "is_correct": is_correct
        })

        total += 1
        correct += int(is_correct)

    accuracy = correct / total if total > 0 else 0

    print(f"✅ Evaluated {total} samples")
    print(f"🎯 Accuracy: {accuracy:.3f} ({correct}/{total})")

    return pd.DataFrame(detailed_results)

if __name__ == "__main__":
    df_eval = evaluate_winoground("results/gpt4o_winoground_online.csv")
    df_eval.to_csv("results/gpt4o_winoground_eval_summary.csv", index=False)
