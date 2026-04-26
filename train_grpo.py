# ==============================================================================
# GOOGLE COLAB — paste these cells in order (GPU runtime: Runtime → Change runtime type → T4 GPU)
#
# --- Cell 1: clone + deps + install this repo as a package (needed for `hospital_triage` imports) ---
# !git clone https://huggingface.co/spaces/HotaroOreki-art/hospital_triage
# %cd /content/hospital_triage
# !pip install -q unsloth trl datasets wandb matplotlib openenv-core
# !pip install -q -e .
#
# --- Cell 2 (optional): Weights & Biases — skip if you want offline-only logs ---
# !wandb login
#
# --- Cell 3: run training (recommended — keeps __file__ correct) ---
# !python train_grpo.py
#
# Or paste this entire file into one cell (also works); Drive mount runs only on Colab.
#
# Notes for beginners:
# - Training uses TASK_SEQUENCE from the live HospitalTriageEnvironment (not train_scenarios.json).
# - data/test_scenarios.json is for offline scenarios; this script does not load it unless you change the code.
# - I cannot push to Hugging Face for you from here; use the “Push your changes” section in the chat reply.
# ==============================================================================

import os
import re
import json
import gc
import torch
import matplotlib

try:
    from google.colab import drive as _colab_drive

    _IN_COLAB = True
except ImportError:
    _colab_drive = None
    _IN_COLAB = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Free any zombie memory from previous Colab crashes (skip if no CUDA — avoids edge-case errors)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# IMPORTANT: Unsloth must be imported before transformers/trl for optimizations to apply
from unsloth import FastLanguageModel
from transformers import TrainerCallback, set_seed
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from hospital_triage.models import HospitalTriageAction
from hospital_triage.server.hospital_triage_environment import HospitalTriageEnvironment, TASK_SEQUENCE

# 1. Reproducibility
set_seed(42)
if _IN_COLAB:
    _colab_drive.mount("/content/drive")

# 2. Model Loading (Unsloth 4-bit for Colab T4 GPU) — before prompts so tokenizer chat template matches the base model
# Context must fit max_prompt_length + max_completion_length (e.g. 2500 + 512)
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Enable PEFT for GRPO
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. Training prompts from live environment scenarios
setup_env = HospitalTriageEnvironment()
train_data = []
for task_name in TASK_SEQUENCE:
    obs = setup_env.reset(task_name=task_name)
    raw_prompt = f"""You are an autonomous triage agent. Given the hospital state, determine the safest action.
State:
{json.dumps(obs.model_dump(), indent=2)}

Respond ONLY with a single valid JSON object (no markdown, no commentary) containing exactly these keys for the OpenEnv hospital triage API:
{{"command": "...", "patient_id": "...", "doctor_id": "...", "room_id": "...", "time_slot": "...", "question": "...", "recommendation_id": "..."}}

Use null or empty string "" for any field that does not apply to the current action.

Rules:
* DO NOT explain anything
* DO NOT output text outside the JSON object
* INVALID format = failure
"""
    messages = [{"role": "user", "content": raw_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    train_data.append({"prompt": prompt, "task_name": task_name})

print(f"✅ Loaded {len(train_data)} training tasks from TASK_SEQUENCE")

dataset = Dataset.from_list(train_data)

# 4. Reward Functions (clinical, environment-grounded)
def clinical_reward_func(completions, task_name, **kwargs):
    """Phase 2: Parse action JSON and score with the live environment."""
    rewards = []
    for completion, t_name in zip(completions, task_name):
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)

        print(f"MODEL OUTPUT: {text[:200]}")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            rewards.append(-1.0)
            continue

        try:
            parsed_json = json.loads(match.group(0))
            action_obj = HospitalTriageAction(**parsed_json)
            
            eval_env = HospitalTriageEnvironment()
            eval_env.reset(task_name=t_name)
            obs = eval_env.step(action_obj)
            
            # The OpenEnv step returns an observation object containing the reward
            rewards.append(obs.reward)
            
        except Exception as e:
            # Catch JSONDecodeError, Pydantic ValidationError, etc.
            rewards.append(-1.0)
            continue

    return rewards

# 5. Custom Callback for Matplotlib Plotting
class MetricPlotterCallback(TrainerCallback):
    """Automatically extracts logs during training and generates hackathon PNG plots."""
    def __init__(self):
        self.losses = []
        self.rewards = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.losses.append(logs["loss"])
                self.steps.append(state.global_step)
            # TRL typically logs reward as 'reward' or similar
            reward_val = logs.get("reward", logs.get("eval_reward", 0))
            self.rewards.append(reward_val) # So reward graph reflects early learning
                
    def on_train_end(self, args, state, control, **kwargs):
        if self.losses:
            plt.figure(figsize=(10, 5))
            plt.plot(self.steps, self.losses, label="Training Loss", color="#ff4c4c", linewidth=2)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Model Training Loss (GRPO)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig("loss.png", dpi=300, bbox_inches="tight")
            print("📈 Saved beautiful loss.png")
            
        if self.rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(self.rewards)), self.rewards, label="Total Reward", color="#4cff4c", linewidth=2)
            plt.xlabel("Log Entry")
            plt.ylabel("Reward")
            plt.title("Model Reward Progression")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig("reward.png", dpi=300, bbox_inches="tight")
            print("📈 Saved beautiful reward.png")

# 6. Inference Demo Function
def run_inference(task_name, description="Inference"):
    print(f"\n{'='*50}\n{description}: {task_name}\n{'='*50}")
    FastLanguageModel.for_inference(model) # Enable native inference speeds
    
    test_prompt = train_data[0]["prompt"]
    _device = next(model.parameters()).device
    inputs = tokenizer([test_prompt], return_tensors="pt").to(_device)
    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Strip the prompt
    answer = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    
    print("\n[Model Output]:")
    print(answer)
    print("\n" + "="*50)

# 7. Training Execution
if __name__ == "__main__":
    print("\n" + "*"*60)
    print("🏥 HACKATHON HOSPITAL TRIAGE TRAINING SCRIPT")
    print("*"*60 + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script expects a GPU runtime (e.g. Colab: Runtime → Change runtime type → GPU)."
        )

    # A. Baseline Demo
    print("🤖 Running Baseline (Untrained) Inference...")
    run_inference("test_task_0", description="Baseline (Untrained)")

    # B. Configure Training (W&B only if a key exists — netrc from `wandb login` counts too)
    _wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes")
    _wandb_usable = False
    if not _wandb_disabled:
        if os.environ.get("WANDB_API_KEY"):
            _wandb_usable = True
        else:
            try:
                from wandb.util import api_key as _wandb_api_key

                _wandb_usable = bool(_wandb_api_key())
            except Exception:
                _wandb_usable = False
    _report_to = "wandb" if _wandb_usable else "none"
    if _report_to == "none":
        print(
            "ℹ️ Using report_to='none'. For W&B: `pip install wandb` then `wandb login`, "
            "or set WANDB_API_KEY. Set WANDB_DISABLED=true to force offline."
        )

    os.environ["WANDB_PROJECT"] = "hospital-triage-rl"

    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2, # MUST be 2 or 4 for Colab T4 to avoid OOM
        max_prompt_length=2500,
        max_completion_length=512,
        logging_steps=1,
        report_to=_report_to,
        run_name="hospital_triage_grpo_run",
        max_steps=50, # Set to a small number for hackathon demonstration speed
    )

    plot_callback = MetricPlotterCallback()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[clinical_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[plot_callback],
    )

    # C. Train
    print("\n🚀 Starting GRPO Training (local loss/reward plots; W&B if configured)...")
    trainer.train()

    # D. Post-Training Demo
    print("\n🧠 Running Trained Inference...")
    run_inference("test_task_0", description="Trained Model")

    # E. Save Adapters (Drive on Colab after mount; local folder otherwise)
    if _IN_COLAB:
        _save_dir = "/content/drive/MyDrive/hospital_triage_phase2_model"
    else:
        _save_dir = os.environ.get("PHASE2_SAVE_DIR", "grpo_hospital_triage_model")
    model.save_pretrained(_save_dir)
    tokenizer.save_pretrained(_save_dir)
    print(f"\n✅ Training complete. Adapters saved to {_save_dir}")
    print("Check the left sidebar for loss.png and reward.png (Colab) or your working directory.")
