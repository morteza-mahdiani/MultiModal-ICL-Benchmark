import pandas as pd
import ace_tools as tools


def winoground_idefics():           
    # Load the provided CSV files
    text_questions_file = "data/idefics2_vismin_evaluation_results.csv"
    image_questions_file = "data/idefics2_vismin_evaluation_results_image_questions.csv"

    df_text = pd.read_csv(text_questions_file)
    df_image = pd.read_csv(image_questions_file)

    # Define the correct answers based on the task description
    correct_text_answers = {0: "A.", 1: "B."}  # Mapping for text questions
    correct_image_answers = {0: "First.", 1: "Second."}  # Mapping for image questions

    # Compute correctness for text and image questions
    df_text["text_correct"] = df_text.apply(lambda row: row["text_answer_1"] == correct_text_answers[0] and 
                                                    row["text_answer_2"] == correct_text_answers[1], axis=1)
    df_image["image_correct"] = df_image.apply(lambda row: row["text_answer_1"] == correct_image_answers[0] and 
                                                        row["text_answer_2"] == correct_image_answers[1], axis=1)

    # Merge with df_text using unique category information
    df2_selected_image = df_image[["image_correct"]]
    df_merged = pd.concat([df_text, df2_selected_image], axis=1)

    # Standardize category names to match Table 2 format
    category_mapping = {
        "object": "Object",
        "attribute": "Attribute",
        "relation": "S. Relation",
        "counting": "Count"
    }

    df_merged["category"] = df_merged["category"].map(category_mapping)

    # Compute per-category scores
    category_scores = {}
    for category in ["Object", "Attribute", "S. Relation", "Count"]:
        df_category = df_merged[df_merged["category"] == category]
        
        text_score = df_category["text_correct"].mean() * 100
        image_score = df_category["image_correct"].mean() * 100
        group_score = (df_category["text_correct"] & df_category["image_correct"]).mean() * 100

        category_scores[category] = [text_score, image_score, group_score]

    # Create the final DataFrame similar to Table 2 in the paper
    winoground_category_results = pd.DataFrame.from_dict(category_scores, orient="index", 
                                                        columns=["Text Score (T)", "Image Score (I)", "Group Score (G)"])

    # Print the results
    print(winoground_category_results)

    # Compute overall average as sum of all values divided by 12 (since 4 categories × 3 metrics)
    overall_avg = winoground_category_results.values.sum() / 12
    print(f"Overall Average: {overall_avg:.2f}")


def winoground_llava():
    # Load the LLaVa evaluation results
    print("LLaVa Winoground Results")


if __name__ == "__main__":
    winoground_idefics()
    # winoground_llava()