# ==============================================================================
# HACKATHON NOTE: Google Colab Setup
# Run the following in your first Colab cell to prepare the environment:
#
# !git clone https://huggingface.co/spaces/HotaroOreki-art/hospital_triage
# %cd hospital_triage
# !pip install -q openenv-core unsloth trl datasets wandb matplotlib
# !wandb login
# ==============================================================================

import os
import re
import json
import gc
import torch
import matplotlib.pyplot as plt

# Free any zombie memory from previous Colab crashes
torch.cuda.empty_cache()
gc.collect()

# IMPORTANT: Unsloth must be imported before transformers/trl for optimizations to apply
from unsloth import FastLanguageModel
from transformers import TrainerCallback, set_seed
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from hospital_triage.server.hospital_triage_environment import HospitalTriageEnvironment, TASK_SEQUENCE

# 1. Reproducibility
set_seed(42)

# 2. Environment Setup
env = HospitalTriageEnvironment()
print(f"✅ Loaded Environment with {len(env._task_map)} Tasks!")

# Prepare Training & Testing Prompts
train_data = []
test_data = []

for task_name in TASK_SEQUENCE:
    obs = env.reset(task_name=task_name)
    prompt = f"""You are an autonomous triage agent. Given the hospital state, determine the safest action.
State:
{json.dumps(obs.model_dump(), indent=2)}

Respond ONLY with valid JSON containing your decision. Example:
{{"command": "SendToER", "patient_id": "p-1-0"}}
"""
    formatted_prompt = [{"role": "user", "content": prompt}]
    if "test" in task_name.lower():
        test_data.append({"prompt": formatted_prompt, "task_name": task_name})
    else:
        train_data.append({"prompt": formatted_prompt, "task_name": task_name})

if not train_data:
    print("⚠️ No valid data found, using dummy data for verification.")
    train_data = [{"prompt": f"State {i}", "task_name": f"task_{i}"} for i in range(10)]

dataset = Dataset.from_list(train_data)

# 3. Model Loading (Unsloth 4-bit for Colab T4 GPU)
max_seq_length = 3000
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2b-it", # Swap with unsloth/Qwen2.5-7B-Instruct for production
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

# 4. Reward Functions (Robust Parsing)
def extract_json(text):
    """Regex helper to reliably extract JSON from LLM outputs."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None

def safety_reward_func(completions, **kwargs):
    """Rewards models that issue SendToER."""
    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion
        parsed = extract_json(text)
        score = 0.0
        if parsed and parsed.get("command") == "SendToER":
            score += 0.5
        rewards.append(score)
    return rewards

def formatting_reward_func(completions, **kwargs):
    """Rewards models that output strictly valid JSON."""
    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion
        parsed = extract_json(text)
        score = 0.5 if parsed and "command" in parsed and "patient_id" in parsed else 0.0
        rewards.append(score)
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
            if reward_val != 0:
                self.rewards.append(reward_val)
                
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
    
    test_prompt = test_data[0]["prompt"] if test_data else train_data[0]["prompt"]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        test_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(inputs, max_new_tokens=256, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Because we used a chat template, the output will contain the whole prompt.
    # A quick way to get the new tokens is to decode only the generated slice.
    generated_ids = outputs[0][inputs.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    print("\n[Model Output]:")
    print(answer)
    print("\n" + "="*50)

# 7. Training Execution
if __name__ == "__main__":
    print("\n" + "*"*60)
    print("🏥 HACKATHON HOSPITAL TRIAGE TRAINING SCRIPT")
    print("*"*60 + "\n")

    # A. Baseline Demo
    print("🤖 Running Baseline (Untrained) Inference...")
    run_inference("test_task_0", description="Baseline (Untrained)")

    # B. Configure Training
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2, # MUST be 2 or 4 for Colab T4 to avoid OOM
        max_prompt_length=2500,
        max_completion_length=256,
        logging_steps=1,
        report_to="wandb", # Full WandB Integration!
        run_name="hospital_triage_grpo_run",
        max_steps=50, # Set to a small number for hackathon demonstration speed
    )

    plot_callback = MetricPlotterCallback()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[safety_reward_func, formatting_reward_func],
        args=training_args,
        train_dataset=dataset,
        callbacks=[plot_callback]
    )

    # C. Train
    print("\n🚀 Starting GRPO Training with WandB and Local Plotting...")
    trainer.train()

    # D. Post-Training Demo
    print("\n🧠 Running Trained Inference...")
    run_inference("test_task_0", description="Trained Model")

    # E. Save Adapters
    model.save_pretrained("grpo_hospital_triage_model")
    tokenizer.save_pretrained("grpo_hospital_triage_model")
    print("\n✅ Training complete. Check the left sidebar for loss.png and reward.png!")
