import subprocess
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from Curriculum_checkpoint_selection import select_curriculum_checkpoint
import weighted_rl_scheduling
import numpy as np

import RL_data_creation_PPO





BASE_MODEL          = "meta-llama/Llama-3.2-3B"
PPO_CHECKPOINT_DIR  = Path("PPO/PPO_Curriculum_Checkpoints")
REWARD_MODELS_DIR = Path("reward_models");
EVALUATION_GENERATIONS_DIR = Path("Evaluations/Generations");
STATE_FILE          = Path("curriculum_state.json")   # tracks last completed iteration
LOG_DIR             = Path("logs")
MAX_ITERS           = 5
STEP_BATCH_SIZE     = 1
ROLLOUT_BATCH_SIZE  = 1
SKIPPED_DATA        = 0
EVAL_BATCH_SIZE     = 32
EVAL_BoN            = 3
ALPACA_PROMPTS_FILE = "data/sampled_alpaca_with_ids&prose.json"
BASE_INDICES        = "0-30000" 
POLICY_INDICES      = "0-30000"






CUES = [
    "structural_distribution",
    "input_ground_overlap",
    "tokens_absolute_length",
    "output_variance",
]






BASE_MU = np.array([
    2.0,   # structural_distribution (strongest)
    0.5,   # input_ground_overlap
    0.2,   # tokens_absolute_length
    0.0    # output_variance (weakest)
])



def sample_weights(mu_vec, sigma=0.5):
    z = np.random.normal(mu_vec, sigma)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def decreasing_dominance_weights(
        current_t=0,
        T=MAX_ITERS,
        sigma=0.2,
    ):


    alpha_t = 1 - current_t / T

    mu_t = alpha_t * BASE_MU

    w = sample_weights(mu_t, sigma)

    weight_dict = dict(zip(CUES, w))

    return weight_dict







PPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True);

LOG_DIR.mkdir(parents=True, exist_ok=True);

REWARD_MODELS_DIR.mkdir(parents=True, exist_ok=True);

EVALUATION_GENERATIONS_DIR.mkdir(parents=True, exist_ok=True);




def setup_logging(iteration: int | None = None) -> logging.Logger:
    """Set up dual logging: one persistent run log + one per-iteration log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("curriculum")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    run_fh = logging.FileHandler(LOG_DIR / f"run_{timestamp}.log")
    run_fh.setFormatter(fmt)
    logger.addHandler(run_fh)

    if iteration is not None:
        iter_fh = logging.FileHandler(LOG_DIR / f"iteration_{iteration}.log")
        iter_fh.setFormatter(fmt)
        logger.addHandler(iter_fh)

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = setup_logging()




def load_state() -> dict:
    """Load curriculum state from disk (last completed step per iteration)."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        logger.info(f"Resuming from saved state: {state}")
        return state
    return {"last_completed_iteration": -1, "last_completed_step": None}


def save_state(iteration: int, step: str) -> None:
    """Persist the last successfully completed (iteration, step) pair."""
    state = {
        "last_completed_iteration": iteration,
        "last_completed_step": step,
        "timestamp": datetime.now().isoformat(),
    }
    # Atomic write: write to a temp file then rename
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)
    logger.debug(f"State saved → iteration={iteration}, step={step}")


def step_already_done(state: dict, iteration: int, step: str) -> bool:
    """Return True if this step was completed in a previous run."""
    STEPS = ["data_gen", "post_processing", "reward_model", "ppo", "eval"]
    last_iter  = state["last_completed_iteration"]
    last_step  = state["last_completed_step"]

    if iteration < last_iter:
        return True
    if iteration == last_iter and last_step is not None:
        return STEPS.index(step) <= STEPS.index(last_step)
    return False


# =========================
# SIGNAL HANDLING (SLURM / HPC)
# =========================

# Turn on if using an HPC

_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.warning(f"Received signal {sig_name} — will exit cleanly after current step.")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_signal)   # SLURM sends SIGTERM before killing
signal.signal(signal.SIGUSR1, _handle_signal)   # --signal USR1@<time> in sbatch


def check_shutdown():
    if _shutdown_requested:
        logger.warning("Shutdown requested. Exiting gracefully.")
        sys.exit(0)



def run_sh(cmd_list: list[str], step_label: str, retries: int = 1) -> None:
    """
    Run a subprocess with optional retry logic and timing.
    Raises RuntimeError if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        logger.info(f"[{step_label}] attempt {attempt}/{retries}: {' '.join(cmd_list)}")
        t0 = time.time()
        result = subprocess.run(cmd_list)
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info(f"[{step_label}] ✅ Done in {elapsed:.1f}s")
            return
        logger.warning(f"[{step_label}] ❌ Failed (exit {result.returncode}) after {elapsed:.1f}s")
        if attempt < retries:
            wait = 30 * attempt
            logger.info(f"[{step_label}] Retrying in {wait}s …")
            time.sleep(wait)

    raise RuntimeError(f"[{step_label}] failed after {retries} attempt(s)")



def get_latest_checkpoint(i:int) -> str:
    ckpts = sorted(PPO_CHECKPOINT_DIR.glob(f"curriculum_iteration_{i}"))
    return str(ckpts[-1]) if ckpts else BASE_MODEL




def reward_model_exists(iteration: int) -> bool:
    return Path(f"reward_models/reward_model_curriculum_iteration_{iteration}").exists()




def ppo_output_exists(iteration: int) -> bool:
    return (PPO_CHECKPOINT_DIR / f"curriculum_iteration_{iteration}").exists()




def log_disk_usage() -> None:
    """Log current disk usage — useful on HPC where quotas are strict."""
    try:
        result = subprocess.run(["df", "-h", "."], capture_output=True, text=True)
        logger.info(f"Disk usage:\n{result.stdout.strip()}")
    except Exception:
        pass




def run_data_generation(
    i: int,
    policy_model: str,
    is_base: bool,
    structural_cues_weights: dict,
    max_distance: int,
    maxDistancePositive: int,
    include_categorical_length_difference:bool,
    include_absolute_length: bool,
    activate_all_structural_distribution: bool,
) -> None:
    base_flags = (
        ["--use_basemodel", "True", "--indices", BASE_INDICES]
        if is_base else
        ["--use_basemodel", "False", "--indices", POLICY_INDICES,
         "--checkpoint_dir", policy_model]
    )
            
    run_sh(
        [
            "bash", "scripts/synthetic_preference_data_generation.sh",
            "--outDir",                       f"data/Synthetic_preference_data/curriculum_iteration_{i}",
            "--promptsFile",                  ALPACA_PROMPTS_FILE,
            "--max_distance",                 str(max_distance),
            "--structural_distribution_weight", str(structural_cues_weights["structural_distribution"]),
            "--tokens_absolute_length_weight",str(structural_cues_weights["tokens_absolute_length"]),
            "--output_variance_weight",       str(structural_cues_weights["output_variance"]),
            "--input_ground_overlap_weight",  str(structural_cues_weights["input_ground_overlap"]),
            "--maxDistancePositive",str(maxDistancePositive),
            "--include_categorical_length_difference",str(include_categorical_length_difference),
            "--include_absolute_length",str(include_absolute_length),
            "--activate_all_structural_distribution", str(activate_all_structural_distribution),
            *base_flags,
        ],
        step_label=f"iter{i}/data_gen",
    )




    
def run_post_processing(i: int, structural_classes_remove: list = None, maxDistancePositive: int = 0.9, demonstration_based: bool = False, min_margin:int = 0.25) -> None:
    cmd = [
        "bash", "scripts/run_post_processing.sh",
        "--in_dir",         f"data/Synthetic_preference_data/curriculum_iteration_{i}/generated_preference_data.jsonl",
        "--out_csv",        f"data/Synthetic_preference_data/curriculum_iteration_{i}/curriculum_iteration_{i}_generated_preference_data.csv",
        "--merged_out_csv", f"data/Synthetic_preference_data/curriculum_iteration_{i}/curriculum_iteration_{i}_merged_balanced_generated_preference_data.csv",
        "--MaxPositiveDistance",str(maxDistancePositive),
        "--demonstration_based",str(demonstration_based),
        "--min_margin", str(min_margin),
    ]

    if structural_classes_remove:
        cmd += ["--structural_classes_remove", str(structural_classes_remove)]

    run_sh(cmd, step_label=f"iter{i}/post_processing");
    








def run_reward_model_training(i: int) -> None:
    dataset = f"data/Synthetic_preference_data/curriculum_iteration_{i}/curriculum_iteration_{i}_merged_balanced_generated_preference_data.csv";
    
    output  = f"reward_models/reward_model_curriculum_iteration_{i}"

    if i > 0 and reward_model_exists(i - 1):
        logger.info(f"Fine-tuning reward model from iteration {i-1} checkpoint.")
    # extra = ["--checkpoint_dir", f"../../base_reward_model/checkpoint50_iteration3_reward_model_alpaca_grounded_prose&non-prose_balanced/checkpoint-2052"]
        extra = ["--checkpoint_dir", f"reward_models/reward_model_curriculum_iteration_{i - 1}"]
    else:
        logger.info("Training reward model from scratch.")
        extra = ["--checkpoint_dir", ""]

    run_sh([
        "bash", "scripts/reward_model_training_script_run.sh",
        "--dataset_path", dataset,
        "--output_dir",   output,
        *extra,
    ], step_label=f"iter{i}/reward_model")











def run_ppo(i: int, policy_model: str, is_base: bool, dataset_path: str) -> None:
    ppo_output = f"PPO/PPO_Curriculum_Checkpoints/curriculum_iteration_{i}"
    common = [
        "--reward_model_name_or_path", f"reward_models/reward_model_curriculum_iteration_{i}",
        "--output_dir",         ppo_output,
        "--resume_from_training","False" if is_base else "True",
        "--skipped_data",       str(SKIPPED_DATA),
        "--step_batch_size",    str(STEP_BATCH_SIZE),
        "--rollout_batch_size", str(ROLLOUT_BATCH_SIZE),
        "--dataset_path",str(dataset_path),
    ];

    ## In case of instability or model collapse in iterations >= 1, reference_model_dir can be turned into the pretrained base-model

    extra = (["--fully_initialize_policy", "True"]
             if is_base else
             # ["--fully_initialize_policy", "False", "--resume_dir", policy_model, "--reference_model_dir",""])
             ["--fully_initialize_policy", "False", "--resume_dir", policy_model, "--reference_model_dir",policy_model])

    run_sh(["bash", "scripts/ppo_training_run.sh", *common, *extra],
           step_label=f"iter{i}/ppo")





def run_evaluation(i: int, policy_model: str, is_base: bool, reward_model_path: str = None,greedy: bool = True, BoN : bool = 1, scheduling_type = "balanced") -> None:
    use_base_flag = "True" if is_base else "False"
    greedy_flag = "True" if greedy else "False"
    for dataset_type in ("humaneval","IFEval","alpaca"):
        print(f"We are now evaluating on {dataset_type}");
        out_file = (f"Evaluations/Generations/curriculum_iteration_{i}_"
                    f"{dataset_type}_eval_generations_{Path(policy_model).name}_{'greedy' if greedy else 'BoN = ' + str(BoN)}_len256_{scheduling_type}.jsonl")
        run_sh([
            "bash", "scripts/run_evaluation_generations.sh",
            "--use_base_model", use_base_flag,
            "--dataset_type",   dataset_type,
            "--outFile",        out_file,
            "--batchSize",      str(EVAL_BATCH_SIZE),
            "--checkpoint_dir", policy_model + "/adapter_model/lora_policy",
            "--bon_n", str(BoN),
            "--greedy",greedy_flag,
            "--reward_model_checkpoint_dir", reward_model_path
        ], step_label=f"iter{i}/eval/{dataset_type}")



    
        
def run_truthfulQA(i: int, policy_model: str, is_base: bool, scheduling_type: str = "balanced") -> None:
    
    use_base_flag = "True" if is_base else "False";
    out_dir = (f"Evaluations/Generations/curriculum_iteration_{i}_"
                f"truthfulQA-mc1-{policy_model}-{scheduling_type}");
    run_sh([
        "bash", "scripts/TruthfulQA_evaluation.sh",
        "--use_base_model", use_base_flag,
        "--outDir",        out_dir,
        "--checkpoint_dir", policy_model + "/adapter_model/lora_policy",
    ], step_label=f"iter{i}/eval/truthfulQA")





def main():
    
    global logger

    state = load_state()
    start_iter = max(0, state["last_completed_iteration"]
                     if state["last_completed_step"] == "eval"
                     else state["last_completed_iteration"])




    logger.info(f"Starting curriculum loop — iterations {start_iter} to {MAX_ITERS - 1}")
    log_disk_usage()

    start_iter = 0;
    
    for i in range(start_iter, MAX_ITERS):
        
        logger = setup_logging(i);
        
        logger.info(f"========== ITERATION {i} ==========")

        check_shutdown()

        policy_model_path = get_latest_checkpoint(i = i-1);
        
        
        is_base = (policy_model_path == BASE_MODEL);
        
        # is_base = False;
        
        # policy_model = "PPO/Llama3.2-3b-ppo-rulebased-level3_checkpoint350_prose&nonprose/ppo-checkpoint-50";
        
        # policy_model = "PPO/PPO_Curriculum_Checkpoints/curriculum_iteration_3/ppo-checkpoint-1250";

        policy_model = select_curriculum_checkpoint(policy_model_path);


        if is_base or policy_model == "base_model":
            policy_model = BASE_MODEL
            is_base = True

            
        logger.info(f"Policy: {policy_model}  (base={is_base})")

        
        weights_dict = decreasing_dominance_weights(current_t = i, T=MAX_ITERS, sigma=0.2);
        
        
        print(f"========= Current iteration Structural cues {weights_dict}============ ");
        
        
        if i == 0:
            ## Should be tuned if needed to make a clear main structural contrast (structural presence >> non-presence)
            maxDistancePositive = 0.5;
            include_categorical_length_difference = False;
            include_absolute_length = False;
            seed = 42
            max_distance = 0.3;
            activate_all_structural_distribution = False;
            scheduling_type = "balanced";
            demonstration_based = False;

        elif i <=2:
            include_categorical_length_difference = True;
            include_absolute_length = False;
            maxDistancePositive = 0.6;
            activate_all_structural_distribution = False
            seed = 42
            max_distance = 0.25
            scheduling_type = "balanced";
            demonstration_based = False;

        else:
            ## activate_all_structural_distribution: To model structural types as a composition of all structures (good for later stages only)
            include_categorical_length_difference = True;
            include_absolute_length = True;
            activate_all_structural_distribution = True
            maxDistancePositive = 0.9;
            seed = 42
            max_distance = 0.15;
            scheduling_type = "balanced";
            demonstration_based = True;



            
        if not step_already_done(state, i, "data_gen"):
            
            run_data_generation(i,policy_model + "/adapter_model/lora_policy", is_base, structural_cues_weights = weights_dict, max_distance = max_distance, maxDistancePositive = maxDistancePositive, include_categorical_length_difference = include_categorical_length_difference, include_absolute_length = include_absolute_length, activate_all_structural_distribution = activate_all_structural_distribution);
            save_state(i, "data_gen")
            state = load_state()
        else:
            logger.info(f"[iter{i}/data_gen] already done — skipping")

        check_shutdown()


        scheduling_weights = weighted_rl_scheduling.main(
            f"data/Synthetic_preference_data/curriculum_iteration_{i}/generated_preference_data.jsonl",
        )
        
        structural_classes_remove = [];
        
        if scheduling_type  == "weighted":
            structural_classes_remove = structural_classes_remove + [k for k, v in scheduling_weights.items() if v == 0];
            


        print(f"=============== Structural classes to remove are ===========", structural_classes_remove);

        
        if not step_already_done(state, i, "post_processing"):
            run_post_processing(i, structural_classes_remove = structural_classes_remove, maxDistancePositive = maxDistancePositive, demonstration_based = demonstration_based, min_margin = max_distance)
            save_state(i, "post_processing")
            state = load_state();

            
        else:
            logger.info(f"[iter{i}/post_processing] already done — skipping")
        check_shutdown();
        

        # ── Step 3: Reward model training ────────────────────────────────
        
        if not step_already_done(state, i, "reward_model"):
            if not reward_model_exists(i):
                run_reward_model_training(i)
            else:
                logger.info(f"Reward model for iteration {i} already exists — skipping training")
            save_state(i, "reward_model")
            state = load_state()
        else:
            logger.info(f"[iter{i}/reward_model] already done — skipping")

        check_shutdown();


        output_file = f"data/curriculum_{i}_PPO_scheduled_{scheduling_type}_training_data.json";
        
        RL_data_creation_PPO.main(scheduling_type = scheduling_type, scheduling_weights = None, output_file = output_file,seed = seed, data_path = f"data/Synthetic_preference_data/curriculum_iteration_{i}/curriculum_iteration_{i}_generated_preference_data.csv",removed_class = "")
        

        if not step_already_done(state, i, "ppo"):
            
            if not ppo_output_exists(i):
                run_ppo(i, policy_model, is_base, dataset_path = output_file)
            else:
                logger.info(f"PPO checkpoint for iteration {i} already exists — skipping")
            save_state(i, "ppo")
            state = load_state()
        else:
            logger.info(f"[iter{i}/ppo] already done — skipping")

        check_shutdown()


        ### Add truthfulQA        
        if not step_already_done(state, i, "eval"):
            ppo_checkpoint = str(PPO_CHECKPOINT_DIR / f"curriculum_iteration_{i}")
            selected_eval_checkpoint = select_curriculum_checkpoint(ppo_checkpoint)
            print(f"++++++ selected_eval_checkpoint {selected_eval_checkpoint} ++++++");
            if selected_eval_checkpoint == "base_model":
                selected_eval_checkpoint = BASE_MODEL
                eval_is_base = True
            else:
                eval_is_base = False
                
            
            run_truthfulQA(i, selected_eval_checkpoint, eval_is_base);
            
            run_evaluation(i, selected_eval_checkpoint, eval_is_base,reward_model_path = "",greedy = True, BoN = 1);
            
            run_evaluation(i, selected_eval_checkpoint, eval_is_base,  reward_model_path = f"reward_models/reward_model_curriculum_iteration_{i}", greedy = False, BoN = EVAL_BoN);
            
            
            
            save_state(i, "eval")
            state = load_state()
        else:
            logger.info(f"[iter{i}/eval] already done — skipping")


        log_disk_usage()
        logger.info(f"========== ITERATION {i} COMPLETE ==========\n")

    logger.info("🎉 All iterations complete.")


if __name__ == "__main__":
    main()