import random
import tqdm
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import fire

# QUESTION TYPE
# creative writing.
# 0: GENERAL (ShareGPT + Dolly)


CHATGPT_LANGUAGES = {
    "it": "Italian",
    "pl": "Polish",
    "ru": "Russian",
    "sk": "Slovak",
    "pt": "Portuguese",
    "ro": "Romanian",
    "da": "Danish",
    "sv": "Swedish",
    "no": "Norwegian",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "cs": "Czech",
    "de": "German",
    "fi": "Finnish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "hu": "Hungarian",
    "he": "Hebrew",
    "el": "Greek",
    "ar": "Arabic",
    "kr": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-traditional": "Chinese (Traditional)",
    "zh-simplified": "Chinese (Simplified)",
    "th": "Thai",
    "vi": "Vietnamese",
}


def remove_leading_fraction(input_string):
    # remove leading fraction
    cleaned_string = re.sub(r"^\d+\s*/\s*\d+", "", input_string)
    cleaned_string = re.sub(r"\d+\s*/\s*\d+$", "", cleaned_string)
    cleaned_string = cleaned_string.split("1 / 1", 1)[-1]

    # \uc9c0\uae08 \ubc88\uc5ed\ud558\uae30
    cleaned_string = cleaned_string.split("지금 번역하기")[0]

    # Language: English
    cleaned_string = cleaned_string.split("\n \n Language: ")[0]

    cleaned_string = cleaned_string.strip()

    if cleaned_string.endswith("Share Prompt"):
        cleaned_string = cleaned_string[: -len("Share Prompt")].strip()

    if cleaned_string.endswith("Translate now"):
        cleaned_string = cleaned_string[: -len("Translate now")].strip()

    for lang_code in CHATGPT_LANGUAGES:
        lang_suffix = f"Language: {CHATGPT_LANGUAGES[lang_code]}"
        if cleaned_string.endswith(lang_suffix):
            cleaned_string = cleaned_string[: -len(lang_suffix)].strip()

    # ~The following is a conversation with Bing, not ChatGPT.~
    if cleaned_string.startswith(
        "~The following is a conversation with Bing, not ChatGPT.~"
    ):
        cleaned_string = cleaned_string[
            len("~The following is a conversation with Bing, not ChatGPT.~") :
        ].strip()

    return cleaned_string







def filter_and_clean_examples(merged_examples):
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")
    max_seq_length = 256
    filtered_examples = []
    set_of_unique_instructions = set()

    # Filter out examples with non-ascci characters
    merged_examples = [
        {
            "instruction": remove_leading_fraction(example["instruction"]),
            "input": example.get("input",""),
            "output": example.get("output",""),
            "length_bonus": example["length_bonus"],
            "question_type": example["question_type"],
            "structural_class":example.get("structural_class",""),
        }
        for example in merged_examples
    ]
    

    for example in tqdm.tqdm(merged_examples):
        instruction = example["instruction"]
        instruction_token_length = len(tokenizer.encode(instruction))
        if (
            2 <= instruction_token_length <= max_seq_length
            and instruction not in set_of_unique_instructions
        ):
            filtered_examples.append(example)
            set_of_unique_instructions.add(instruction)

    return filtered_examples

    


def load_json(share_gpt_data_path):
    with open(share_gpt_data_path, "r") as f:
        share_gpt_data = json.load(f)
    examples = []
    for data in share_gpt_data:
        examples.append(
            {
                "instruction": data["instruction"],
                "input": "",
                "output": "",
            }
        )
    print(f"{share_gpt_data_path} examples:", len(examples))
    return examples







def load_attaq_data(length_bonus):
    
    dataset = load_dataset("ibm-research/AttaQ")["train"];
    
    category_to_examples = {}
    
    for example in dataset:
        category = example["label"]
        if category not in category_to_examples:
            category_to_examples[category] = []
        category_to_examples[category].append(example)

    merged_examples = []
    for data in [
        category_to_examples["violence"],
        category_to_examples["discrimination"],
        category_to_examples["substance_abuse"],
        category_to_examples["deception"],
        category_to_examples["harmful_info"],
        category_to_examples["pii"],
        category_to_examples["explicit_content"],
    ]:
        for i in range(len(data)):
            merged_examples.append(
                {
                    "instruction": data[i]["input"],
                    "input": "",
                    "output": "",
                    "length_bonus": length_bonus,
                    "question_type": 1,
                }
            )
    print("AttaQ examples:", len(merged_examples))
    return merged_examples





def load_salad_data(length_bonus):
    
    dataset = load_dataset("OpenSafetyLab/Salad-Data", "base_set")["train"];
    
    category_to_examples = {}
    
    for example in dataset:
        category = example["1-category"];
        if category not in category_to_examples:
            category_to_examples[category] = [];
        category_to_examples[category].append(example)

    merged_examples = []
    for data in [
        category_to_examples["O2: Misinformation Harms"],
        category_to_examples["O5: Malicious Use"],
        category_to_examples["O4: Information & Safety"],
        category_to_examples["O3: Socioeconomic Harms"],
        category_to_examples["O1: Representation & Toxicity"],
        category_to_examples["O6: Human Autonomy & Integrity"],
    ]:
        for i in range(len(data)):
            merged_examples.append(
                {
                    "instruction": data[i]["question"],
                    "input": "",
                    "output": "",
                    "length_bonus": length_bonus,
                    "question_type": 1,
                    "dimension":data[i]["1-category"]
                }
            )
    print("Salad examples:", len(merged_examples))
    return merged_examples









def load_do_not_answer_data(length_bonus):
    
    dataset = load_dataset("LibrAI/do-not-answer")["train"];
    
    category_to_examples = {}
    
    for example in dataset:
        category = example["types_of_harm"]
        if category not in category_to_examples:
            category_to_examples[category] = []
        category_to_examples[category].append(example)

    # print(category_to_examples);
    merged_examples = []

    
    for data in [
        category_to_examples["Adult Content"],
        category_to_examples["Social stereotypes and unfair discrimination"],
        category_to_examples["Toxic language (hate speech)"],
        category_to_examples["Mental Health or Overreliance Crisis"],
        category_to_examples["Treat Chatbot as a Human"],
        category_to_examples["Compromise privacy by leaking or inferring private information (person/individual)"],
        category_to_examples["Risks from leaking or inferring sensitive information (organization/gov)"],
        category_to_examples["Assisting illegal activities"],
        category_to_examples["Nudging or advising users to perform unethical or unsafe actions"],
        category_to_examples["Reducing the cost of disinformation campaigns"],
        category_to_examples["Causing material harm by disseminating misinformation e.g. in medicine or law"],
        category_to_examples["Disseminating false or misleading information"],
    ]:
        for i in range(len(data)):
            merged_examples.append(
                {
                    "instruction": data[i]["question"],
                    "input": "",
                    "output": "",
                    "length_bonus": length_bonus,
                    "question_type": 2,
                }
            )
    
    print("Do not ask examples:", len(merged_examples))
    return merged_examples;








def load_Nvidia_steer_helpfulness_data(length_bonus):
    dataset = load_dataset("nvidia/HelpSteer")["train"];
    merged_examples = [];
    for example in dataset:
        merged_examples.append(
            {
                "instruction": example["prompt"],
                "input": "",
                "output": "",
                "length_bonus": length_bonus,
                "question_type": 0,
            }
        )
    print("HELP steer ask examples:", len(merged_examples))
    return merged_examples;


    

def load_pku_data(length_bonus):
    
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")['train'];
    merged_examples = [];
    for example in dataset:
        merged_examples.append(
            {
                "instruction": example["prompt"],
                "input": "",
                "output": "",
                "length_bonus": length_bonus,
                "question_type": 1,
            }
        )
        
    print("HELP steer ask examples:", len(merged_examples))
    return merged_examples;











import numpy as np

def alpaca_scheduling_samples(length_bonus, seed=42, json_path=None, 
                              sampling_probs=None, mode="balanced",
                              max_samples=None, removed_class="table", 
                              selected_structures=[],
                              concatenate_lists=False):


    LIST_CLASSES = {"Dashed-list", "Numbered-list", "Comma-list"}

    if json_path is None:
        json_path = "komondoro_test/Komondor codebase/data/sampled_alpaca_with_ids&prose.json"

    ext = os.path.splitext(json_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(json_path)
    elif ext == ".json":
        df = pd.read_json(json_path)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Expected .json or .csv")
    print(f"Loaded {len(df)} rows from {json_path}")

    # Rename columns if needed
    if "class" in df.columns and "structural_class" not in df.columns:
        df = df.rename(columns={"class": "structural_class"})
    if "ground_truth" in df.columns:
        df = df.rename(columns={"ground_truth": "output"})

    required_cols = {"instruction", "input", "structural_class", "output"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"File is missing required columns: {missing}")

    df = df.drop_duplicates()
    df = df.fillna({"input": ""})
    df = df.fillna({"output": ""})
    print(f"Total rows after dedup: {len(df)}")

    if removed_class:
        df = df[~df["structural_class"].str.lower().str.contains(removed_class)]
    if selected_structures:
        print(f"selected_structures are {selected_structures}=======")
        df = df[df["structural_class"].isin(selected_structures)]

    if df.empty:
        raise ValueError("No samples left after filtering structural classes.")

    if mode == "balanced":
        if concatenate_lists:
            # Split into list pool (all three subtypes merged) and non-list classes
            list_df = df[df["structural_class"].isin(LIST_CLASSES)]
            non_list_df = df[~df["structural_class"].isin(LIST_CLASSES)]

            # Find the per-slot count: min across non-list classes and the merged list pool
            non_list_counts = non_list_df["structural_class"].value_counts()
            min_count = min(non_list_counts.min(), len(list_df))

            list_sample = list_df.sample(min_count, random_state=seed).reset_index(drop=True)

            non_list_sample = (
                non_list_df.groupby("structural_class", group_keys=False)
                           .apply(lambda x: x.sample(min_count, random_state=seed))
                           .reset_index(drop=True)
            )

            grouped = {
                cls: group.sample(frac=1, random_state=seed).reset_index(drop=True)
                for cls, group in non_list_sample.groupby("structural_class")
            }
            grouped["list"] = list_sample.sample(frac=1, random_state=seed).reset_index(drop=True)

        else:
            min_count = df["structural_class"].value_counts().min()
            balanced_df = (
                df.groupby("structural_class", group_keys=False)
                  .apply(lambda x: x.sample(min_count, random_state=seed))
                  .reset_index(drop=True)
            )
            grouped = {
                cls: group.sample(frac=1, random_state=seed).reset_index(drop=True)
                for cls, group in balanced_df.groupby("structural_class")
            }

        classes = list(grouped.keys())
        random.Random(seed).shuffle(classes)
        interleaved_rows = []
        for rows in zip(*[grouped[cls].itertuples(index=False) for cls in classes]):
            interleaved_rows.extend(rows)
        if max_samples is not None:
            interleaved_rows = interleaved_rows[:max_samples]

    elif mode == "weighted":
        if sampling_probs is None:
            raise ValueError("sampling_probs must be provided in weighted mode.")

        total_prob = sum(sampling_probs.values())
        probs = {k: v / total_prob for k, v in sampling_probs.items()}

        grouped = {cls: group for cls, group in df.groupby("structural_class") if cls in probs}
        if not grouped:
            raise ValueError("No structural classes match the provided sampling_probs.")

        rng = np.random.default_rng(seed)
        total_samples = sum(len(g) for g in grouped.values())
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)

        classes_list = list(grouped.keys())
        class_sizes = {cls: len(g) for cls, g in grouped.items()}
        chosen_classes = rng.choice(classes_list, size=total_samples, p=[probs[c] for c in classes_list])

        class_indices = {cls: 0 for cls in classes_list}
        interleaved_rows = []
        for cls in chosen_classes:
            idx = class_indices[cls]
            if idx < class_sizes[cls]:
                interleaved_rows.append(grouped[cls].iloc[idx])
                class_indices[cls] += 1

    else:
        raise ValueError("mode must be 'balanced' or 'weighted'")

    merged_examples = [
        {
            "instruction": row.instruction,
            "input": row.input if row.input else "",
            "output": row.output if row.output else "",
            "length_bonus": length_bonus,
            "question_type": 1,
            "structural_class": row.structural_class,
        }
        for row in interleaved_rows
    ]
    print(merged_examples[:5])
    print(f"Total generated examples ({mode} mode): {len(merged_examples)}")
    return merged_examples

































def load_Nvidia_steer_helpfulness2_data(length_bonus):
    # Load the dataset from a JSON file
    dataset = load_dataset("nvidia/HelpSteer2")["train"]
    
    merged_examples = []
    for example in dataset:
        merged_examples.append(
            {
                "instruction": example["prompt"],  # adjust this if your JSON field name is different
                "input": "",
                "output": "",
                "length_bonus": length_bonus,
                "question_type": 0,
            }
        )
    
    print("HELP steer2 ask examples:", len(merged_examples))
    return merged_examples




import pandas as pd

import os

def main(
    sharegpt_prompt_path: str = "data/RL_data/writing_share_gpt.json",
    output_file: str = "data/PPO_scheduled_training_data.json",
    data_path:str = "data/sampled_alpaca_with_ids&prose.json",
    scheduling_type: str = "balanced", # Balanced or weighted 
    scheduling_weights: dict = None, # Balanced or weighted 
    max_samples: int = 2000,
    general_length_bonus: float = 0.5,
    red_teaming_length_penalty: float = 1.5, 
    do_not_answer_penalty: float = -0.5, 
    seed :int = 42,
    selected_structures: list = [],
    removed_class:str = "",
):


    if scheduling_weights and scheduling_type == "weighted":
        alpaca = alpaca_scheduling_samples(length_bonus = general_length_bonus, seed=seed, json_path=data_path, 
                                  sampling_probs = scheduling_weights, mode=scheduling_type,
                                  max_samples = max_samples, selected_structures = selected_structures, removed_class = removed_class);
    else:
        alpaca = alpaca_scheduling_samples(length_bonus = general_length_bonus, seed=seed, json_path=data_path, 
                                  sampling_probs = None, mode ="balanced" ,
                                  max_samples = max_samples,
                                          removed_class = removed_class, selected_structures= selected_structures);
        

    merged_examples = []
    merged_examples.extend(alpaca)

    filtered_examples = filter_and_clean_examples(merged_examples)

    

    print("Total examples:", len(filtered_examples))

    with open(output_file, "w") as f:
        json.dump(filtered_examples, f, indent=2);





if __name__ == "__main__":
    fire.Fire(main)