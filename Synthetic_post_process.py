import csv
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional, Union

import fire
import pandas as pd


_FIELDNAMES = [
    "instruction",
    "input",
    "chosen",
    "rejected",
    "preference",
    "contrastive_status",
    "structural_class",
    "positive_distance",
    "negative_distance",
    "ground_truth",         # ground-truth / full_output from generation
    "demonstration_based",  # True when the row was created by demo substitution
    "id",
]




DEMONSTRATION_DISTANCE_THRESHOLD: float = 0.3


def _iter_records(in_path: Path, start_idx: int, end_idx: int):

    if in_path.is_file():
        # Single .jsonl file
        with in_path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    datum = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Use _idx tag if present, else fall back to line position
                idx = datum.get("_idx", line_no)
                if not (start_idx <= idx < end_idx):
                    continue

                yield datum

    elif in_path.is_dir():
        # Directory of numbered .json shards
        for path in sorted(in_path.glob("*.json")):
            try:
                file_idx = int(path.stem)
            except ValueError:
                continue

            if not (start_idx <= file_idx < end_idx):
                continue

            with path.open("r", encoding="utf-8") as rf:
                for raw in rf:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        datum = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    yield datum

    else:
        raise FileNotFoundError(f"in_dir/in_file not found: {in_path}")



def json_to_csv(
    in_dir: Union[str, Path],
    out_csv: Union[str, Path],
    start_idx: int = 0,
    end_idx: int = 10**12,
) -> None:
    """Read records from in_dir (directory of .json shards or a .jsonl file)
    and write valid rows to out_csv.

    Only records whose status field is truthy are emitted.
    """
    in_path = Path(in_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()

        for datum in _iter_records(in_path, start_idx, end_idx):
            if not datum.get("status", False):
                continue

            writer.writerow(
                {
                    "instruction":        datum["prompt"].strip(),
                    "input":              datum["input"].strip(),
                    "chosen":             datum["chosen_result"].strip(),
                    "rejected":           datum["rejected_result"].strip(),
                    "preference":         datum.get("preference", ""),
                    "contrastive_status": datum.get("contrastive_status", ""),
                    "structural_class":   datum.get("structural_class", ""),
                    "positive_distance":  datum.get("positive_distance", ""),
                    "negative_distance":  datum.get("negative_distance", ""),
                    "ground_truth":       datum.get("ground_truth", ""),
                    "demonstration_based": datum.get("demonstration_based", False),
                    "id":                 datum.get("id", ""),
                }
            )
            written += 1

    print(f"[json_to_csv] wrote {written} rows -> {out_csv}")



def clean(
    source: Union[str, Path, pd.DataFrame],
    min_margin: float = 0.2,
    swap: bool = True,
    MaxPositiveDistance: int = 0.9,
    demonstration_based: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()

    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""
    if "demonstration_based" not in df.columns:
        df["demonstration_based"] = False

    assert set(_FIELDNAMES).issubset(set(df.columns)), (
        f"Missing columns: {set(_FIELDNAMES) - set(df.columns)}"
    )

    keep_mask = pd.Series(True, index=df.index)
    demo_rows: list[dict] = [] 
    

    for i, row in df.iterrows():

        pos, neg = row["positive_distance"], row["negative_distance"]
        ground_truth = str(row.get("ground_truth", "")).strip()

        if demonstration_based and ground_truth and pd.notna(pos) and pd.notna(neg):

            if (
                pos <= DEMONSTRATION_DISTANCE_THRESHOLD
                and ground_truth.strip() != str(row["chosen"]).strip()
            ):
                
                demo_rows.append({
                    "instruction":         row["instruction"],
                    "input":               row["input"],
                    "chosen":              ground_truth,
                    "rejected":            str(row["chosen"]).strip(),
                    "preference":          row.get("preference", ""),
                    "contrastive_status":  "Contrastive",
                    "structural_class":    row["structural_class"],
                    "positive_distance":   0.0,
                    "negative_distance":   pos,
                    "ground_truth":        ground_truth,
                    "demonstration_based": True,
                    "id":                  row["id"],
                })

            if (
                neg <= DEMONSTRATION_DISTANCE_THRESHOLD
                and ground_truth.strip() != str(row["rejected"]).strip()
            ):
                demo_rows.append({
                    "instruction":         row["instruction"],
                    "input":               row["input"],
                    "chosen":              ground_truth,
                    "rejected":            str(row["rejected"]).strip(),
                    "preference":          row.get("preference", ""),
                    "contrastive_status":  "Contrastive",
                    "structural_class":    row["structural_class"],
                    "positive_distance":   0.0,
                    "negative_distance":   neg,
                    "ground_truth":        ground_truth,
                    "demonstration_based": True,
                    "id":                  row["id"],
                })

        if pd.isna(pos) or pd.isna(neg) or pos >= MaxPositiveDistance:
            keep_mask[i] = False
            continue

        rejected_is_verbatim = row["rejected"] in (row["instruction"], row["input"]);
        
        chosen_is_verbatim   = row["chosen"]  in (row["instruction"], row["input"])

        if swap and neg <= pos - min_margin and not rejected_is_verbatim and not str(row["rejected"]).strip() == str(row["chosen"]).strip() and not df.at[i, "contrastive_status"] == "Contrastive":
            df.at[i, "chosen"],            df.at[i, "rejected"]           = row["rejected"], row["chosen"]
            df.at[i, "positive_distance"], df.at[i, "negative_distance"]  = neg, pos
            df.at[i, "contrastive_status"] = "Contrastive"

        
        elif (
            pos <= neg - min_margin
            and df.at[i, "contrastive_status"] == "Contrastive"
            and not chosen_is_verbatim
            and not str(row["rejected"]).strip() == str(row["chosen"]).strip()
        ):
            print("VALID & PASSED: ",pos, neg);
            pass 

        elif (
            pos <= neg - min_margin
            and df.at[i, "contrastive_status"] == "Non-contrastive"
            and not chosen_is_verbatim
            and not str(row["rejected"]).strip() == str(row["chosen"]).strip()
        ):
            print("PASSED:",pos, neg);
            df.at[i, "contrastive_status"] = "Contrastive"

                
        else:
            keep_mask[i] = False;

            

    result = df[keep_mask].reset_index(drop=True)

    if demo_rows:
        demo_df = pd.DataFrame(demo_rows, columns=_FIELDNAMES)
        result = pd.concat([result, demo_df], ignore_index=True)
        print(f"[demonstration_based] appended {len(demo_rows)} new demonstration rows "
              f"({sum(1 for r in demo_rows if r['negative_distance'] == r.get('negative_distance') and r['positive_distance'] == 0.0)} total)")

    return result






def merge(
    base: pd.DataFrame,
    extra_csv: Union[str, Path],
    min_margin: float = 0.2,
) -> pd.DataFrame:
    
    extra = clean(extra_csv, min_margin=min_margin, swap=True)
    return pd.concat([base, extra], ignore_index=True)



def _row_pn(row) -> tuple:
    d = row._asdict()
    return (
        d.get("chosen", ""),
        d.get("rejected", ""),
        d.get("positive_distance"),
        d.get("negative_distance"),
    )


def combine_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    synthetic = []

    for entry_id, group in df.groupby("id"):
        if len(group) < 2:
            continue

        rows = list(group.itertuples(index=False))
        for r1, r2 in itertools.permutations(rows, 2):
            pos1, neg1, pd1, nd1 = _row_pn(r1)
            pos2, neg2, pd2, nd2 = _row_pn(r2)

            base = {
                "id":                 entry_id,
                "instruction":        r1.instruction,
                "input":              r1.input,
                "preference":         1,
                "contrastive_status": "Contrastive",
            }

            synthetic.append({
                **base,
                "chosen":            pos1,
                "rejected":          neg2,
                "structural_class":  r1.structural_class,
                "positive_distance": pd1,
                "negative_distance": nd2,
            })
            synthetic.append({
                **base,
                "chosen":            pos2,
                "rejected":          neg1,
                "structural_class":  r2.structural_class,
                "positive_distance": pd2,
                "negative_distance": nd1,
            })

    if not synthetic:
        return df

    combined = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
    combined.drop_duplicates(inplace=True)
    return combined



def balance(df: pd.DataFrame, max_per_class: int = 1000, seed: int = 42, balancing_rate: int = 1.5) -> pd.DataFrame:

    print("Initial DataFrame shape:", df.shape)
    df = df.copy()

    before = len(df)
    df = df[df["structural_class"] != "List"]
    print(f"Removed {before - len(df)} rows with structural_class == 'list'")

    list_variants = {"dashed-list", "numbered-list"}
    # list_variants = {"dashed-list", "numbered-list","comma-list"}
    df["structural_class"] = df["structural_class"].apply(
        lambda x: "list" if isinstance(x, str) and x.lower() in list_variants else x
    )
    print("Class distribution AFTER merging 'dashed-list' and 'numbered-list':")
    print(df["structural_class"].value_counts())

    grouped = df.groupby("structural_class")

    min_size = min(len(grp) for _, grp in grouped)
    target_size = min(int(min_size * balancing_rate), max_per_class)
    print(f"Smallest class size: {min_size}")
    print(f"Target size per class: {target_size}")

    frames = []
    print("Sampling per class:")

    for cls, grp in grouped:
        replacement = len(grp) < target_size
        print(f"  → Class '{cls}': original={len(grp)} → sampled={target_size}")
        sampled = grp.sample(n=target_size, random_state=seed, replace=replacement)
        frames.append(sampled)

    balanced_df = (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    print("Final class distribution:")
    print(balanced_df["structural_class"].value_counts())

    return balanced_df



def finalise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.rename(columns={"chosen": "output_1", "rejected": "output_2"})[
        ["output_1", "output_2", "preference", "instruction", "input"]
    ].copy()

    swap_mask = pd.Series([rng.random() < 0.5 for _ in range(len(out))], index=out.index)
    out.loc[swap_mask, ["output_1", "output_2"]] = (
        out.loc[swap_mask, ["output_2", "output_1"]].values
    )
    out["preference"] = swap_mask.map({True: 2, False: 1})
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out



def main(
    in_dir: str = "",
    out_csv: str = "",
    merged_out_csv: str = "",
    start_idx: int = 0,
    end_idx: int = 10**12,
    data_to_merge: str = "",
    structural_classes_remove: list = None,
    do_balance: bool = True,
    max_per_class: int = 30000,
    do_synthetic: bool = False,
    seed: int = 42,
    MaxPositiveDistance: int = 0.6,
    demonstration_based: bool = False,
    min_margin:int = 0.25,
) -> None:

    random.seed(seed)

    print("in_dir", in_dir)
    json_to_csv(in_dir, out_csv, start_idx, end_idx)
    print(pd.read_csv(out_csv).head())

    df = clean(
        out_csv,
        min_margin=min_margin,
        swap=True,
        MaxPositiveDistance=MaxPositiveDistance,
        demonstration_based=demonstration_based,
    )
    print(f"[clean] {len(df)} rows after cleaning")

    if data_to_merge:
        sources = [data_to_merge] if isinstance(data_to_merge, str) else data_to_merge
        for src in filter(None, sources):
            df = merge(df, src)

    
    if structural_classes_remove:
        before = len(df)
        df = df[~df["structural_class"].isin(structural_classes_remove)].reset_index(drop=True)
        print(f"[filter] structural_class removed {before - len(df)} rows → {len(df)} remaining")

    if do_balance:
        df = balance(df, max_per_class=max_per_class, seed=seed)
        print(f"[balance] {len(df)} rows after balancing")

    final = finalise(df, seed=seed)

    out_path = Path(merged_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] saved {len(final)} rows -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)