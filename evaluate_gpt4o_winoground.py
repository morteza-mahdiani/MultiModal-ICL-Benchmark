import pandas as pd
from datasets import load_dataset


def normalize_answer(ans):
    if not isinstance(ans, str) or ans.strip() == "":
        return None
    ans = ans.lower().strip()
    if ans.startswith("yes"):
        return True
    if ans.startswith("no"):
        return False
    return None


def extract_tag_type(row):
    try:
        return row["collapsed_tag"]
    except:
        return "unknown"


def evaluate_winoground(results_path, output_detailed="data/winoground_detailed_8shot.csv",
                        output_summary="data/winoground_summary_8shot.csv"):
    # Load model output results
    df = pd.read_csv(results_path)
    df["normalized"] = df["answer"].apply(normalize_answer)

    # Load original dataset with metadata
    ds = load_dataset("facebook/winoground", split="test")
    meta_df = pd.DataFrame(ds)
    print(meta_df.head(5))
    meta_df["sample_id"] = meta_df.index
    meta_df["category"] = meta_df.apply(extract_tag_type, axis=1)

    # Evaluate each sample
    grouped = df.groupby("sample_id")
    results = []

    for sample_id, group in grouped:
        if len(group) != 4:
            continue

        lookup = {
            (row["image"], str(row["caption"])): row["normalized"]
            for _, row in group.iterrows()
        }

        if any(lookup.get(k) is None for k in [('A', '1'), ('A', '2'), ('B', '1'), ('B', '2')]):
            continue

        is_correct = (
                lookup[('A', '1')] is True and
                lookup[('B', '2')] is True and
                lookup[('A', '2')] is False and
                lookup[('B', '1')] is False
        )

        image_score = (
                              (lookup[('A', '1')] is True and lookup[('A', '2')] is False) +
                              (lookup[('B', '2')] is True and lookup[('B', '1')] is False)
                      ) / 2

        text_score = (
                             (lookup[('A', '1')] is True and lookup[('B', '1')] is False) +
                             (lookup[('B', '2')] is True and lookup[('A', '2')] is False)
                     ) / 2

        results.append({
            "sample_id": sample_id,
            "is_correct": is_correct,
            "image_score": image_score,
            "text_score": text_score
        })

    # Combine with metadata
    results_df = pd.DataFrame(results)
    merged = pd.merge(results_df, meta_df[["sample_id", "category"]], on="sample_id")

    # Save full results
    merged.to_csv(output_detailed, index=False)

    # Aggregate summary by category
    summary = merged.groupby("category").agg(
        samples=("sample_id", "count"),
        accuracy=("is_correct", "mean"),
        image_score=("image_score", "mean"),
        text_score=("text_score", "mean")
    ).round(4).reset_index()

    summary.to_csv(output_summary, index=False)
    print("✅ Evaluation complete. Summary:")
    print(summary)
    print(f"\n💾 Saved detailed results to: {output_detailed}")
    print(f"💾 Saved summary to: {output_summary}")


if __name__ == "__main__":
    evaluate_winoground("data/gpt4o_winoground_online_8shot_final_RGB.csv")
