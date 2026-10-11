import json
import os
import random
import csv
import fire
import re
import itertools
import pandas as pd




def from_json_files_to_csv(IN_DIR, OUT_CSV, START_IDX, END_IDX):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as wf:
        fieldnames = [
            'instruction',
            'input',
            'chosen',
            'rejected',
            'preference',
            'contrastive_status',
            'structural_class',
            'positive_distance',
            'negative_distance',
            'id',
        ]
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()

        for fname in os.listdir(IN_DIR):
            if not fname.endswith('.json'):
                continue

            try:
                file_idx = int(fname[:-len('.json')])
            except ValueError:
                continue

            if file_idx < START_IDX or file_idx >= END_IDX:
                continue

            with open(os.path.join(IN_DIR, fname), 'r', encoding='utf-8') as rf:
                for line in rf:

                    datum = json.loads(line)

                    if not datum.get('status', False):
                        continue

                    instruction = datum['prompt'].strip()
                    input_ = datum['input'].strip()


                    chosen = datum['chosen_result'].strip()
                    rejected = datum['rejected_result'].strip()

                    contrastive_status = datum.get('contrastive_status', '')
                    structural_class = datum.get('structural_class', '')
                    sample_id = datum.get('id', '')
                    positive_distance = datum.get('positive_distance', '')
                    negative_distance = datum.get('negative_distance', '')
                    preference = datum.get('preference', '')

                    writer.writerow({
                        'instruction': instruction,
                        'input': input_,
                        'chosen': chosen,
                        'rejected': rejected,
                        'preference': preference,
                        'contrastive_status': contrastive_status,
                        'structural_class': structural_class,
                        'positive_distance': positive_distance,
                        'negative_distance': negative_distance,
                        'id': sample_id,
                    })






def clean_non_contrastive(csv_file, min_margin, swap):
    datum_frame = pd.read_csv(csv_file)
    assert datum_frame.columns.tolist() == [
        'instruction',
        'input',
        'chosen',
        'rejected',
        'preference',
        'contrastive_status',
        'structural_class',
        'positive_distance',
        'negative_distance',
        'id',
    ], "CSV columns do not match expected format."

    rows_to_drop = []

    for i, row in datum_frame.iterrows():
        pos = row['positive_distance']
        neg = row['negative_distance']

        # Skip rows with missing distance values
        if pd.isna(pos) or pd.isna(neg):
            rows_to_drop.append(i)
            continue

        if swap and neg <= pos - min_margin and (row['rejected'] != row['instruction']) and (row['rejected'] != row['input']):
            # Swap chosen/rejected and their distances, mark as Contrastive
            datum_frame.at[i, 'chosen'], datum_frame.at[i, 'rejected'] = row['rejected'], row['chosen']
            datum_frame.at[i, 'contrastive_status'] = 'Contrastive'
            datum_frame.at[i, 'positive_distance'] = neg
            datum_frame.at[i, 'negative_distance'] = pos;
            
        elif pos <= neg + 0.1 and datum_frame.at[i, 'contrastive_status'] == 'Contrastive' and (row['chosen'] != row['instruction']) and (row['chosen'] != row['input']):
            # Already correctly ordered with sufficient margin — keep as-is
            datum_frame.at[i, 'contrastive_status'] = 'Contrastive';
            
        else:
            # Margin too small or wrong direction without swap — remove
            rows_to_drop.append(i)

    datum_frame = datum_frame.drop(index=rows_to_drop).reset_index(drop=True)
    return datum_frame




    


def merge_data(cleaned_pd_datum, data_to_merge_csv):
    # Check if datum to merge is clean
    cleaned_to_merge = clean_non_contrastive(
        csv_file=data_to_merge_csv, min_margin=0.2, swap=True
    )
    print("Length of cleaned to merge", len(cleaned_to_merge));
    print(cleaned_to_merge);

    # Fix: initialize merged_pd from the two cleaned frames
    merged_pd = pd.concat([cleaned_pd_datum, cleaned_to_merge], ignore_index=True)
    return merged_pd


def balanced_data(merged_pd):
    balanced_frames = []

    for structural_class, group in merged_pd.groupby('structural_class'):
        balanced_group = group.sample(n=min(1000, len(group)), random_state=42)
        balanced_frames.append(balanced_group)

    # Fix: initialize balanced_pd properly instead of concatenating onto undefined var
    balanced_pd = pd.concat(balanced_frames, ignore_index=True)
    return balanced_pd


def get_positive_negative(row_dict):
    """Extract the preferred (positive) and non-preferred (negative) outputs
    along with their distances from a row dictionary.
    Assumes 'chosen' is the preferred output and 'rejected' is the non-preferred."""
    pos = row_dict.get('chosen', '')
    neg = row_dict.get('rejected', '')
    pd_ = row_dict.get('positive_distance', None)
    nd_ = row_dict.get('negative_distance', None)
    return pos, neg, pd_, nd_


def combine_synthetic(df):
    """For each group of rows sharing the same 'id', generate synthetic preference
    pairs by mixing positives and negatives across all ordered pairs of rows."""
    synthetic_rows = []

    for entry_id, group in df.groupby('id'):
        if len(group) < 2:
            continue  # need at least two entries to mix

        rows = list(group.itertuples(index=False))

        # All ordered pairs (i, j) where i ≠ j
        for r1, r2 in itertools.permutations(rows, 2):

            pos1, neg1, pd1, nd1 = get_positive_negative(r1._asdict())
            pos2, neg2, pd2, nd2 = get_positive_negative(r2._asdict())

            # ── Combination A: positive from r1, negative from r2 ──────────────
            combo_a = {
                'id':                 entry_id,
                'instruction':        r1.instruction,
                'input':              r1.input,
                'chosen':             pos1,
                'rejected':           neg2,
                'preference':         1,
                'contrastive_status': 'Contrastive',
                'structural_class':   r1.structural_class,
                'positive_distance':  pd1,
                'negative_distance':  nd2,
                'source_row_1':       r1.id,
                'source_row_2':       r2.id,
                'synthetic':          True,
            }

            # ── Combination B: positive from r2, negative from r1 ──────────────
            combo_b = {
                **combo_a,
                'chosen':             pos2,
                'rejected':           neg1,
                'structural_class':   r2.structural_class,
                'positive_distance':  pd2,
                'negative_distance':  nd1,
                'source_row_1':       r2.id,
                'source_row_2':       r1.id,
            }

            synthetic_rows.extend([combo_a, combo_b])

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True);

    return combined_df


def main(
    IN_DIR="../data/Checkpoint_350_alpaca_preference_structural_rule_based_only_prose_structures_withids_only_prose",
    OUT_CSV="../data/Checkpoint_350_alpaca_preference_structural_rule_based_only_prose_structures_withids_only_prose.csv",
    START_IDX=0,
    END_IDX=10**12,
    # MERGED_OUT_CSV="",
    MERGED_OUT_CSV="../data/Checkpoint_350_alpaca_preference_structural_rule_based_only_prose_structures_withids_output_not_merged.csv",
    Data_to_merge= "",
    # Data_to_merge= "../data/Checkpoint_350_alpaca_preference_structural_rule_based_only_prose_structures_withids.csv",
    balance_data = False,
):
    # Step 1: Convert JSON files to CSV
    from_json_files_to_csv(IN_DIR, OUT_CSV, START_IDX, END_IDX)

    print(pd.read_csv(OUT_CSV))

    # Step 2: Clean the resulting CSV
    cleaned_pd_datum = clean_non_contrastive(csv_file=OUT_CSV, swap=True, min_margin=0.1)

    print("cleaned pd datum length :", len(cleaned_pd_datum))
    # Step 3: Optionally merge with additional datasets
    if Data_to_merge:
        merge_sources = [Data_to_merge] if isinstance(Data_to_merge, str) else Data_to_merge
        merge_sources = [s for s in merge_sources if s]  # filter empty strings

        merged_pd = cleaned_pd_datum
        for source in merge_sources:
            merged_pd = merge_data(merged_pd, source)
    else:
        merged_pd = cleaned_pd_datum

    # Step 4: Generate synthetic combinations
    # combined_pd = combine_synthetic(merged_pd)
    combined_pd = merged_pd
    combined_pd = combined_pd[combined_pd['structural_class'] == "Prose"] 
    print("combined pd datum length :", len(combined_pd))


    
    # Step 5: Balance merged dataframe
    if balance_data:
        balanced_merged_pd = balanced_data(combined_pd)
    else:
        balanced_merged_pd = combined_pd;
    
    
    # Step 6: Transform to final output schema
    final_pd = balanced_merged_pd.rename(columns={
        'chosen':   'output_1',
        'rejected': 'output_2',
    })[['output_1', 'output_2', 'preference', 'instruction', 'input']]
    

    mask = pd.Series([random.random() < 0.5 for _ in range(len(final_pd))], index=final_pd.index)
    final_pd.loc[mask, ['output_1', 'output_2']] = final_pd.loc[mask, ['output_2', 'output_1']].values
    final_pd['preference'] = mask.map({True: 2, False: 1})
    
    # Step 7: Save final dataframe
    final_pd.to_csv(MERGED_OUT_CSV, index=False, encoding='utf-8')
    print(f"Saved {len(final_pd)} rows to {MERGED_OUT_CSV}")


if __name__ == "__main__":
    fire.Fire(main)