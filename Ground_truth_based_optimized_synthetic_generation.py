from argparse import Namespace
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import tqdm
import json
import random
import pandas as pd
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import fire
import os
from collections import defaultdict
from models.qlora_model import load_4bit_model_for_inference


from rule_based_contrastive_sampling.utils import *



HF_TOKEN = os.getenv("HF_TOKEN");
login(token=HF_TOKEN);



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
);



positive_aligning_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.

{RuleBasedInfo}

### Instruction:
{Input}{Context}
### Response:{RevealedOutput}
"""




negative_aligning_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.

{RuleBasedInfo}

### Instruction:
{Input}{Context}
### Response:
"""



def _parse_indices_helper(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)


def parse_indices(indices):
    return list(_parse_indices_helper(indices))


def process_filter_result(result):
   
    if "#" in result:
        result = result[:result.index('#')].strip();
        
    # if "```" in result:
    #     result = result.replace("```", "").strip();
        
                
        
    # if not result or result == "":
    if not result or result == "":
        return False, None


    return True, result.strip();




def generate_result(model, tokenizer, prompts, max_new_tokens=256):
    
    if type(prompts) == str:
        prompts = [prompts];
        
        
    bad_words = ['\\', '\\\\', '`', '```']
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    
    all_results = [None for _ in range(len(prompts))]
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda');
        
        # results = model.generate(**inputs, temperature=0.7, do_sample=True, max_new_tokens=max_new_tokens);
        results = model.generate(**inputs, temperature=0.7, do_sample=True, max_new_tokens=max_new_tokens, bad_words_ids = bad_words_ids);
        
        for i in range(len(prompts)):
            result = results[i][len(inputs['input_ids'][i]):]
            result = tokenizer.decode(result, skip_special_tokens=True)
            status, result = process_filter_result(result)
            all_results[i] = (status, result)

            
    return all_results









STRUCTURAL_DISTRIBUTION_WEIGHT: float = 0.7
TOKENS_ABSOLUTE_LENGTH_WEIGHT:  float = 0.1
OUTPUT_VARIANCE_WEIGHT:         float = 0.1
INPUT_GROUND_OVERLAP_WEIGHT:    float = 0.1
MAX_DISTANCE:                   float = 0.25
MIN_MARGIN:                     float = 0.05 
INCLUDE_CATEGORICAL_LENGTH_DIFFERENCE: bool = True
INCLUDE_ABSOLUTE_LENGTH: bool = True
ALL_STRUCTURAL_DISTRIBUTION: bool = False







def set_global_weights(
    structural_distribution_weight: float,
    tokens_absolute_length_weight:  float,
    output_variance_weight:         float,
    input_ground_overlap_weight:    float,
    max_distance:                   float,
    include_categorical_length_difference: bool,
    include_absolute_length: bool,
    activate_all_structural_distribution:bool,
) -> None:
    """Overwrite module-level constants from CLI args."""
    
    global STRUCTURAL_DISTRIBUTION_WEIGHT, TOKENS_ABSOLUTE_LENGTH_WEIGHT
    global OUTPUT_VARIANCE_WEIGHT, INPUT_GROUND_OVERLAP_WEIGHT, MAX_DISTANCE
    global INCLUDE_ABSOLUTE_LENGTH, ALL_STRUCTURAL_DISTRIBUTION, INCLUDE_CATEGORICAL_LENGTH_DIFFERENCE

    
    STRUCTURAL_DISTRIBUTION_WEIGHT = structural_distribution_weight
    TOKENS_ABSOLUTE_LENGTH_WEIGHT  = tokens_absolute_length_weight
    OUTPUT_VARIANCE_WEIGHT         = output_variance_weight
    INPUT_GROUND_OVERLAP_WEIGHT    = input_ground_overlap_weight
    MAX_DISTANCE                   = max_distance
    INCLUDE_CATEGORICAL_LENGTH_DIFFERENCE = include_categorical_length_difference
    INCLUDE_ABSOLUTE_LENGTH = include_absolute_length
    ALL_STRUCTURAL_DISTRIBUTION = activate_all_structural_distribution







    




def _compute_margin(
    dist_fn,
    full_output:     str,
    positive_sample: str,
    text:            str,
    input_context:   str,
) -> tuple[float, bool, float, float]:
    
    ref_dist      = dist_fn(full_output)
    positive_dist = dist_fn(positive_sample)
    neg_dist      = dist_fn(text)

    pos_structural_sim = structure_distance(ref_dist, positive_dist)
    neg_structural_sim = structure_distance(ref_dist, neg_dist)

    pos_distance = _weighted_distance(pos_structural_sim, full_output, positive_sample, input_context, full_output)
    neg_distance = _weighted_distance(neg_structural_sim, full_output, text,            input_context, full_output)

    margin = neg_distance - pos_distance
    print(f"pos_distance {pos_distance:.4f}, neg_distance {neg_distance:.4f}")
    # BUG FIX: was `margin < -min_margin` where min_margin was undefined
    return margin, margin < -MAX_DISTANCE, pos_distance, neg_distance









_DIST_FN_MAP = {
    "Mathematics": maths_distribution_modeling,
    "Code":        code_distribution_modeling,
    "Dialog":      dialogue_distribution_modeling,
    "Questions":   question_distribution_modeling,
    "Number":      number_distribution_modeling,
    "Table":      table_distribution_modeling,
}








def _weighted_distance(
    structural_distance: float,
    full_output: str,
    text: str,
    input_context: str,
    reference: str,          # = full_output for positive; positive_sample for negative
) -> float:
    return (
        STRUCTURAL_DISTRIBUTION_WEIGHT * structural_distance
        + INPUT_GROUND_OVERLAP_WEIGHT  * input_ground_overlap_distance(input_context, full_output, text)
        + OUTPUT_VARIANCE_WEIGHT       * output_variance_distance(full_output, text)
        + TOKENS_ABSOLUTE_LENGTH_WEIGHT* tokens_absolute_length_difference(text, full_output)
    )



    


def calculate_distance(text: str, reference_text: str, structural_class: str) -> float:
    
    if structural_class == "Prose":

        
        keys = [
            prose_distribution_modeling(text),
            prose_distribution_modeling(reference_text),
        ]
        
        gen_dists  = keys[0][:] 
        ref_dists  = keys[1][:];
        
        return sum(
            structure_distance(g, r, include_absolute_length=INCLUDE_ABSOLUTE_LENGTH,include_categorical_length_difference=INCLUDE_CATEGORICAL_LENGTH_DIFFERENCE)
            for g, r in zip(gen_dists, ref_dists)
        ) / len(gen_dists)
        

    dist_fn = _DIST_FN_MAP.get(structural_class);
    
    if dist_fn is None:
        return 0.0
        
    return structure_distance(dist_fn(text), dist_fn(reference_text))






_TRIVIAL_REJECT = (False, False, 1.0, 1.0)
_TRIVIAL_ZERO   = (False, False, 0.0, 0.0)





def _is_trivial(text: str, instruction_only: str, input_only: str) -> bool:
    return text == instruction_only or text == input_only






    
def validate_rule_based_info(
    text:                str,
    rule_based_info:     list,
    structural_class:    str,
    is_positive:         bool  = True,
    revealed_output:     str   = "",
    full_output:         str   = "",
    positive_sample:     str   = "",
    max_distance_threshold: float = 0.7,
    min_distance_threshold: float = 0.9,
    input_context:       str   = "",
    input_only:          str   = "",
    instruction_only:    str   = "",
) -> tuple[bool, bool, float, float]:

    
    has_table = detect_table(text);

    
    trivial   = _is_trivial(text, instruction_only, input_only)


    
    def _pos_distance(structural_dist: float) -> float:
        return _weighted_distance(structural_dist, full_output, text, input_context, full_output)


    
    def _list_dist_fn(sample: str) -> float:
        sc = "Numbered-list" if structural_class == "List" else structural_class
        return list_distribution_modeling(sample, list_type=sc)
        
        

    if is_positive:
        
        distance = 1.0  

        for rule in rule_based_info:
            
            rl = rule.lower()

            if structural_class == "Prose" or ALL_STRUCTURAL_DISTRIBUTION and not structural_class == "Table":
                
                print(f"Structural_class {structural_class} and ALL_STRUCTURAL_DISTRIBUTION {ALL_STRUCTURAL_DISTRIBUTION}");
                
                distance = _pos_distance(calculate_distance(text, full_output, "Prose"));
                
                if distance > max_distance_threshold or trivial:
                    return False, False, distance, 1.0


            
            elif "list" in rl:
                sc       = "Numbered-list" if structural_class == "List" else structural_class
                gt_dist  = list_distribution_modeling(full_output, list_type=sc)
                txt_dist = list_distribution_modeling(text,        list_type=sc)
                structural_dist = structure_distance(txt_dist, gt_dist)
                distance = _pos_distance(structural_dist)
                if distance > max_distance_threshold or trivial:
                    return False, False, distance, 1.0


            
            elif any(kw in rl for kw in ("mathematical", "code", "dialogue", "question", "number", "table")):
                distance = _pos_distance(calculate_distance(text, full_output, structural_class));
                # if distance > max_distance_threshold or trivial or not _PRESENCE_FN_MAP[structural_class](text):
                if distance > max_distance_threshold or trivial:
                    return False, False, distance, 1.0

        
        
        return True, False, distance, 1.0


    pos_distance, neg_distance = 1.0, 1.0

    for rule in rule_based_info:
        rl = rule.lower()

        # if "table" in rl:
        #     if has_table:
        #         return False, False, 0.0, 0.0

        if "list" in rl:
            if not positive_sample:
                return *_TRIVIAL_ZERO,   # (False, False, 0, 0)

            ref_dist      = _list_dist_fn(full_output)
            positive_dist = _list_dist_fn(positive_sample)
            neg_dist      = _list_dist_fn(text)

            pos_distance = _weighted_distance(
                structure_distance(ref_dist, positive_dist),
                full_output, positive_sample, input_context, full_output,
            )
            neg_distance = _weighted_distance(
                structure_distance(ref_dist, neg_dist),
                full_output, text, input_context, full_output,
            )
            margin = neg_distance - pos_distance

            if abs(margin) < MAX_DISTANCE:
                return False, False, pos_distance, neg_distance
            return True, margin < -MAX_DISTANCE and not trivial, pos_distance, neg_distance

        elif structural_class == "Prose":
            
            if not positive_sample:
                return *_TRIVIAL_REJECT,

            # BUG FIX: pos/neg_distance were read before assignment in original
            prose_ref  = calculate_distance(positive_sample, full_output, "Prose")
            prose_neg  = calculate_distance(text,            full_output, "Prose")
            pos_distance = _weighted_distance(prose_ref, full_output, positive_sample, input_context, full_output)
            neg_distance = _weighted_distance(prose_neg, full_output, text,            input_context, full_output)

            print(f"Prose neg_distance={neg_distance:.4f}  pos_distance={pos_distance:.4f}")
            margin = neg_distance - pos_distance
            if abs(margin) < MAX_DISTANCE or trivial:
                return False, False, pos_distance, neg_distance
            return True, margin < -MAX_DISTANCE and not trivial, pos_distance, neg_distance

        
        else:
            # Mathematical / Code / Dialog / Questions / Number / Table
            dist_fn = _DIST_FN_MAP.get(structural_class);
            
            if dist_fn is None:
                continue;
                
            if not positive_sample:
                return *_TRIVIAL_ZERO,

            margin, should_swap, pos_distance, neg_distance = _compute_margin(
                dist_fn, full_output, positive_sample, text, input_context
            )

            if abs(margin) < MAX_DISTANCE:
                return False, False, pos_distance, neg_distance
                
            return True, margin < -MAX_DISTANCE and not trivial, pos_distance, neg_distance
            

    return True, False, pos_distance, neg_distance   # BUG FIX: was UnboundLocalError risk




def create_prompts(
    args,
    instruction:      str,
    input_context:    str,
    output:           str,
    structural_class: str,
    revealing_rate:   float = 0,
    positive_sample:  str   = "",
):
    context = f"\n\n{input_context}" if input_context.strip() else ""

    rule_based_info = []
    _rule_map = {
        "List":         "include any list of multiple elements.",
        "Dashed-list":  "include a dashed list of elements using -.",
        "Numbered-list":"include a numbered list of elements.",
        "Comma-list":   "include a comma list of elements separated by commas (,).",
        "Table":        "include a table structure separated by | .",
        "Questions":    "be a question and contain a question mark (?) .",
        "Mathematics":  "include mathematical formulas, expressions or equations.",
        "Code":         "include code.",
        "Number":       "be a number only.",
        "Dialog":       "include a dialogue between two or more speakers using (:) .",
        "Prose":        "provide a correct answer that fully satisfies the instructions, written as plain prose only.",
    }
    if structural_class in _rule_map:
        rule_based_info.append(_rule_map[structural_class])

    revealed_output = ""
    if output and output.strip() and revealing_rate > 0:
        words        = output.split()
        reveal_len   = max(1, int(len(words) * revealing_rate))
        revealed_output = " ".join(words[:reveal_len])

    if positive_sample:
        # STAGE 2: negative prompt – all variants were identical, collapsed to one
        negative_prompt = negative_aligning_prompt.format(
            Input=instruction, Context=context, RuleBasedInfo=""
        )
        return ["rejected"], [negative_prompt], revealed_output, rule_based_info, structural_class

    # STAGE 1: positive prompt
    rule_text = f"\n The response should {rule_based_info[0]}\n" if rule_based_info else ""
    positive_prompt = positive_aligning_prompt.format(
        Input=instruction,
        Context=context,
        RuleBasedInfo=rule_text,
        RevealedOutput=revealed_output,   # empty string is fine
    )
    return ["chosen"], [positive_prompt], revealed_output, rule_based_info, structural_class






    



def ready_for_analysis(args, analysis_waiting_entry):
    return all([key in analysis_waiting_entry for key in ['chosen', 'rejected']])


def analyze_results(args, batch, model, tokenizer):
    results = []
    for entry in batch:

        if entry.data['revealed_output']:
            chosen_full = entry.data['revealed_output'] + " " + entry.data['chosen']
        else:
            chosen_full = entry.data['chosen']
        
        rejected_full = entry.data['rejected'];
        
        results.append({
            'prompt': entry.data.get('prompt', ''),
            'input': entry.data.get('input', ''),
            'actual_prompts': entry.data.get('actual_prompts', []),
            'chosen_result': chosen_full.strip(),
            'rejected_result': rejected_full.strip(),
            'rule_based_info': entry.data.get('rule_based_info', []),
            'structural_class': entry.data.get('structural_class', 'Unknown'),
            'contrastive_status': entry.data.get('contrastive_status', 'Contrastive'),
            'id': entry.data.get('id', None),
            'ground_truth': entry.data.get('ground_truth', ''),
            'positive_distance': entry.data.get('positive_distance', None),
            'negative_distance': entry.data.get('negative_distance', None),
            'status': True
        })
    return results




def load_completed_indices(jsonl_path):
    """Return the set of integer indices already saved in *jsonl_path*.

    Each line is expected to be a JSON object that was written by
    ``append_result_to_jsonl``.  Lines that are missing an ``_idx`` field or
    that cannot be parsed are silently skipped so that a partially-written
    file never blocks a resume.
    """
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if '_idx' in obj:
                    completed.add(obj['_idx'])
            except json.JSONDecodeError:
                pass
    return completed






def append_result_to_jsonl(jsonl_path, idx, save_dict):
    """Append a single result dict (augmented with ``_idx``) to *jsonl_path*.

    The file is opened in append mode so concurrent / resumed runs never
    overwrite earlier results.
    """
    record = {'_idx': idx, **save_dict}
    with open(jsonl_path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + '\n')





class GenerationQueueEntry:
    def __init__(self, idx, key, retries, prompt, original_prompt, revealed_output, structural_class,
                 rule_based_info = None, require_validation=False, full_output="", input_context="",
                 positive_sample="", positive_sample_index=None, sample_id = 0):
        
        self.idx = idx
        self.key = key
        self.retries = retries
        self.prompt = prompt
        self.original_prompt = original_prompt
        self.revealed_output = revealed_output
        self.structural_class = structural_class
        self.rule_based_info = rule_based_info or []
        self.require_validation = require_validation
        self.full_output = full_output
        self.input_context = input_context
        self.positive_sample = positive_sample
        self.positive_sample_batch_index = positive_sample_index
        self.sample_id = sample_id



class AnalysisQueueEntry:
    def __init__(self, idx, data, revealed_output, rule_based_info, structural_class):
        self.idx = idx
        self.data = data
        self.data['revealed_output'] = revealed_output
        self.data['rule_based_info'] = rule_based_info
        self.data['structural_class'] = structural_class
        self.data['ground_truth'] = self.data.get('ground_truth', '')
        self.data['id'] = self.data.get('id', self.data.get('sample_id', None))
        self.data['positive_distance'] = self.data.get('positive_distance', None)
        self.data['negative_distance'] = self.data.get('negative_distance', None)




def main(
    response_len:                   int   = 256,
    resampling_number:              int   = 5,
    use_basemodel:                  bool  = False,
    checkpoint_dir:                 str   = "",
    promptsFile:                    str   = "data/sampled_alpaca_with_ids&prose.json",
    indices:                        str   = None,
    indicesMod:                     int   = 1,
    indicesRemainder:               int   = 0,
    batchSize:                      int   = 64,
    maxRetries:                     int   = 3,
    outDir:                         str   = "",
    seed:                           int   = 0,
    modelString:                    str   = "meta-llama/Llama-3.2-3B",
    maxResamplingAttempts:          int   = 3,
    maxDistancePositive:          float = 0.9,
    minDistanceNegative:          float = 0,
    structural_distribution_weight: float = STRUCTURAL_DISTRIBUTION_WEIGHT,
    tokens_absolute_length_weight:  float = TOKENS_ABSOLUTE_LENGTH_WEIGHT,
    output_variance_weight:         float = OUTPUT_VARIANCE_WEIGHT,
    input_ground_overlap_weight:    float = INPUT_GROUND_OVERLAP_WEIGHT,
    max_distance:                   float = MAX_DISTANCE,
    include_categorical_length_difference:float = INCLUDE_CATEGORICAL_LENGTH_DIFFERENCE,
    include_absolute_length:float = INCLUDE_ABSOLUTE_LENGTH,
    activate_all_structural_distribution:bool = ALL_STRUCTURAL_DISTRIBUTION,
):
    
    set_global_weights(
        structural_distribution_weight,
        tokens_absolute_length_weight,
        output_variance_weight,
        input_ground_overlap_weight,
        max_distance,
        include_categorical_length_difference = include_categorical_length_difference,
        include_absolute_length = include_absolute_length,
        activate_all_structural_distribution = activate_all_structural_distribution,
    )

    args = Namespace(
        response_len          = response_len,
        resampling_number     = resampling_number,
        use_basemodel         = use_basemodel,
        checkpoint_dir        = checkpoint_dir,
        promptsFile           = promptsFile,
        indices               = indices,
        indicesMod            = indicesMod,
        indicesRemainder      = indicesRemainder,
        batchSize             = batchSize,
        maxRetries            = maxRetries,
        outDir                = outDir,
        seed                  = seed,
        modelString           = modelString,
        maxResamplingAttempts = maxResamplingAttempts,
        maxDistancePostive = maxDistancePositive,
        minDistanceNegative = minDistanceNegative,
    )



    from collections import defaultdict
    import random
    import json
    
    rng = random.Random(args.seed)
    
    with open(args.promptsFile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    class_groups = defaultdict(list)
    
    for item in data:
        cls = item.get('class', 'Prose')
        if cls.lower() not in ["list"]:
        # if cls.lower() in ["prose"]:
            class_groups[cls].append(item);
    
    if not class_groups:
        raise ValueError("No classes found after filtering.")
        
    max_size = max(len(v) for v in class_groups.values());

    max_size = min(max_size, 10000);

    balanced_items = []
    
    for cls, items in class_groups.items():
        if len(items) < max_size:
            sampled = [rng.choice(items) for _ in range(max_size)]
        elif len(items) > max_size:
            sampled = rng.sample(items, max_size)
        else:
            sampled = items.copy()
        
        print(cls, len(items), "→", len(sampled))
        balanced_items.extend(sampled)

    data_entries = []



    for item in balanced_items:
        for _ in range(10):
            data_entries.append({
                'instruction': item['instruction'].strip(),
                'input': item.get('input', '').strip(),
                'output': item.get('output', '').strip(),
                'class': item.get('class', 'Prose'),
                'id': item['id']
            })
            
    
    rng.shuffle(data_entries)
    print("++++++++Data entries length: ++++++++++",len(data_entries));


    os.makedirs(args.outDir, exist_ok=True)

    jsonl_path = os.path.join(args.outDir, 'generated_preference_data.jsonl');
    completed_indices = load_completed_indices(jsonl_path);
    print(f"[resume] {len(completed_indices)} indices already saved in {jsonl_path}");

    
    tokenizer = AutoTokenizer.from_pretrained(args.modelString, padding_side='left')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token


    
    if args.use_basemodel:
        
        model = AutoModelForCausalLM.from_pretrained(
            args.modelString, 
            device_map='auto', 
            quantization_config=bnb_config
        ).eval();

        
    else:
        model = load_4bit_model_for_inference(
            checkpoint_dir=args.checkpoint_dir,
            bits=4,
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            adapter_name="lora_policy",
            is_trainable=False,
            reuse_base_model=True,
            trust_remote_code=True,
            base_model_mapping=None,
            fully_initialize=False,
            base_model_name_or_path_for_fully_initialize=args.modelString,
        )

    
    indices_list = parse_indices(args.indices) if args.indices is not None else list(range(len(data_entries)))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            
            current_idx = -1
            failed_indices = set()
            generation_queue = []
            analysis_waiting = defaultdict(lambda: {})
            analysis_queue = []
            resampling_attempts = defaultdict(lambda: {'chosen': 0, 'rejected': 0})

            while current_idx < len(data_entries) or len(generation_queue) > 0 or len(analysis_queue) > 0:
              

                if len(analysis_queue) >= args.batchSize or (len(analysis_queue) > 0 and current_idx >= len(data_entries)):
                    
                    batch = analysis_queue[:args.batchSize]
                    analysis_queue = analysis_queue[args.batchSize:]
                    assert all([x.idx not in failed_indices for x in batch])
                    
                    save_dicts = analyze_results(args, batch, model, tokenizer)
                    
                    for entry, save_dict in zip(batch, save_dicts):
                        append_result_to_jsonl(jsonl_path, entry.idx, save_dict)
                        if entry.idx in resampling_attempts:
                            del resampling_attempts[entry.idx]


                elif len(generation_queue) >= args.batchSize or (len(generation_queue) > 0 and current_idx >= len(data_entries)):
                    
                    batch = generation_queue[:args.batchSize]
                    generation_queue = generation_queue[args.batchSize:]
                    batch = [x for x in batch if x.idx not in failed_indices]
                    batch_prompts = [x.prompt for x in batch]
                    
                    batch_results = generate_result(model, tokenizer, batch_prompts, max_new_tokens=args.response_len)                    
                    
                    for i, (status, result) in enumerate(batch_results):
                     
                        if status:  
                            
                            is_positive = (batch[i].key == 'chosen')

                        
                            is_valid, should_swap, positive_distance, negative_distance = validate_rule_based_info(
                                result, 
                                batch[i].rule_based_info,
                                batch[i].structural_class,
                                is_positive=is_positive,
                                revealed_output=batch[i].revealed_output,
                                full_output=batch[i].full_output,
                                positive_sample=batch[i].positive_sample,
                                max_distance_threshold=args.maxDistancePostive,
                                min_distance_threshold=args.minDistanceNegative,
                                input_context=batch[i].original_prompt + " " + (batch[i].input_context if batch[i].input_context else ""),
                                input_only = batch[i].input_context if batch[i].input_context else "",
                                instruction_only = batch[i].original_prompt,
                            )
                            
                            is_plausible = True;
                            perplexity = None;


                            
                            if is_positive:
                                
                                should_accept = is_valid;
                                
                                if not should_accept:
                                    
                                    resampling_attempts[batch[i].idx][batch[i].key] += 1;
                                    
                                    if resampling_attempts[batch[i].idx][batch[i].key] < args.maxResamplingAttempts:
                                        generation_queue.append(batch[i])
                                        continue;
                                        
                                    else:
                                        analysis_waiting[batch[i].idx]['contrastive_status'] = 'Non-contrastive'

                                        
                                
                                analysis_waiting[batch[i].idx][batch[i].key] = result;
                                analysis_waiting[batch[i].idx]['prompt'] = batch[i].original_prompt
                                analysis_waiting[batch[i].idx]['input'] = batch[i].input_context
                                analysis_waiting[batch[i].idx]['ground_truth'] = batch[i].full_output
                                analysis_waiting[batch[i].idx]['id'] = batch[i].sample_id
                                analysis_waiting[batch[i].idx]['positive_distance'] = positive_distance
                                analysis_waiting[batch[i].idx]['negative_distance'] = analysis_waiting[batch[i].idx].get('negative_distance', None)

                                # print(analysis_waiting[batch[i].idx])
                                
                                if 'actual_prompts' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['actual_prompts'] = []
                                    
                                    
                                analysis_waiting[batch[i].idx]['actual_prompts'].append(batch[i].prompt)
                                
                                if 'contrastive_status' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['contrastive_status'] = 'Contrastive';
                                    

                                instruction = batch[i].original_prompt
                                input_context = batch[i].input_context
                                structural_class = batch[i].structural_class
                                
                                if batch[i].revealed_output:
                                    positive_sample_full = batch[i].revealed_output + " " + result
                                else:
                                    positive_sample_full = result
                                
                                keys, new_prompts, revealed_output, rule_based_info, structural_class = create_prompts(
                                    args, instruction, input_context, batch[i].full_output, structural_class = structural_class, 
                                    positive_sample=positive_sample_full
                                );
                                
                                for key, new_prompt in zip(keys, new_prompts):
                                    
                                    generation_queue.append(GenerationQueueEntry(
                                        batch[i].idx, key, 0, new_prompt, instruction, revealed_output,
                                        structural_class=structural_class,
                                        rule_based_info=rule_based_info,
                                        require_validation=True,
                                        full_output=batch[i].full_output,
                                        input_context=input_context,
                                        positive_sample=positive_sample_full
                                    ))

                                    
                                    
                            else:
                                
                                should_accept = is_valid;
                                
                                if not should_accept:
                                    
                                    resampling_attempts[batch[i].idx][batch[i].key] += 1;
                                    
                                    if resampling_attempts[batch[i].idx][batch[i].key] < args.maxResamplingAttempts:
                                        generation_queue.append(batch[i]);
                                        continue
                                        
                                    else:
                                        print(f"Warning: Max resampling attempts reached for idx {batch[i].idx} "
                                              f"(rejected/{batch[i].structural_class}) - couldn't achieve distance ≥ "
                                              f"{args.minDistanceNegative} to positive sample. Marking as Non-contrastive.")
                                        
                                        analysis_waiting[batch[i].idx]['contrastive_status'] = 'Non-contrastive'
                                        
                                
                                if should_swap and 'chosen' in analysis_waiting[batch[i].idx]:
                                    print(f"[idx {batch[i].idx}] Swapping chosen/rejected labels "
                                          f"(negative sample is more ref-like than positive sample).")

                                    analysis_waiting[batch[i].idx]['negative_distance'] = negative_distance
                                    analysis_waiting[batch[i].idx]['positive_distance'] = positive_distance
                                    analysis_waiting[batch[i].idx]['rejected'] = result

                                    # print("before swapping",analysis_waiting[batch[i].idx]);
                                    
                                    old_chosen = analysis_waiting[batch[i].idx].pop('chosen')
                                    analysis_waiting[batch[i].idx]['chosen'] = result
                                    analysis_waiting[batch[i].idx]['rejected'] = old_chosen

                                    old_pos_dist = analysis_waiting[batch[i].idx].pop('positive_distance', None)
                                    old_neg_dist = analysis_waiting[batch[i].idx].pop('negative_distance', None)
                                    if old_pos_dist is not None:
                                        analysis_waiting[batch[i].idx]['negative_distance'] = positive_distance
                                    analysis_waiting[batch[i].idx]['positive_distance'] = negative_distance

                                else:
                                    analysis_waiting[batch[i].idx]['rejected'] = result
                                    analysis_waiting[batch[i].idx]['negative_distance'] = negative_distance
                                    analysis_waiting[batch[i].idx]['positive_distance'] = positive_distance

                                if 'actual_prompts' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['actual_prompts'] = []
                                analysis_waiting[batch[i].idx]['actual_prompts'].append(batch[i].prompt)
                                
                                if perplexity is not None:
                                    if 'rejected_perplexity' not in analysis_waiting[batch[i].idx]:
                                        analysis_waiting[batch[i].idx]['rejected_perplexity'] = perplexity
                                
                                if 'contrastive_status' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['contrastive_status'] = 'Contrastive'
                                
                                if ready_for_analysis(args, analysis_waiting[batch[i].idx]):
                                    analysis_queue.append(AnalysisQueueEntry(
                                        batch[i].idx, 
                                        analysis_waiting[batch[i].idx], 
                                        batch[i].revealed_output, 
                                        batch[i].rule_based_info, 
                                        batch[i].structural_class,   
                                    ))
                                    del analysis_waiting[batch[i].idx]

                        else:
                            
                            batch[i].retries += 1
                            
                            if batch[i].retries >= args.maxRetries:
                                failed_indices.add(batch[i].idx)
                                # ── CHANGED: append failure record to JSONL ──────────────
                                append_result_to_jsonl(jsonl_path, batch[i].idx, {'status': False})
                                # ─────────────────────────────────────────────────────────
                                if batch[i].idx in analysis_waiting:
                                    del analysis_waiting[batch[i].idx]
                                if batch[i].idx in resampling_attempts:
                                    del resampling_attempts[batch[i].idx]
                            else:
                                generation_queue.append(batch[i])
                else:
                    
                    current_idx += 1;
                    
                    if current_idx >= len(data_entries) or current_idx not in indices_list or current_idx % args.indicesMod != args.indicesRemainder:
                        continue
                        
                    if current_idx in completed_indices:
                        continue

                    # print(f"++++++++++++ Current_idx passed the barrier {current_idx} +++++++++ \n\n")
                    
                    instruction    = data_entries[current_idx]['instruction']
                    input_context  = data_entries[current_idx]['input']
                    output         = data_entries[current_idx]['output']
                    structural_class = data_entries[current_idx]['class']
                    sample_id = data_entries[current_idx]['id']
                    
                    keys, new_prompts, revealed_output, rule_based_info, structural_class = create_prompts(
                        args, instruction, input_context, output, structural_class = structural_class
                    )
                    
                    require_validation = True
                    
                    for key, new_prompt in zip(keys, new_prompts):
                        generation_queue.append(GenerationQueueEntry(
                            current_idx, key, 0, new_prompt, instruction, revealed_output,
                            structural_class=structural_class,
                            rule_based_info=rule_based_info,
                            require_validation=require_validation,
                            full_output=output,
                            input_context=input_context,
                            positive_sample="",
                            sample_id = sample_id
                        ))
                        


if __name__ == "__main__":
    fire.Fire(main)