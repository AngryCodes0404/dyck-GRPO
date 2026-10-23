from dataclasses import dataclass, field
from typing import Dict, List, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

from dyck_dataset import load_dyck_json
from reward import dyck_reward


@dataclass
class ScriptArguments:
    model_name: str = field(
        default="./qwen3-8b-local",
        metadata={"help": "Path or name of base model to finetune with GRPO"},
    )
    dataset_path: str = field(
        default="./lgc_v2_challenge_40000.json",
        metadata={"help": "Path to Dyck language JSON dataset"},
    )
    output_dir: str = field(
        default="./qwen3-8b-dyck-grpo",
        metadata={"help": "Where to save the finetuned model"},
    )
    max_prompt_length: int = 512
    max_gen_length: int = 128
    num_train_steps: int = 2000
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 200
    num_generations_per_prompt: int = 8


def build_prompt(example):
    q = example["prompt"]
    return (
        q.strip()
        + "\n\n"
        + "Answer with only the completed sequence, with no explanation."
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    ds = load_dyck_json(script_args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    def map_to_rl(example):
        return {
            "prompt": build_prompt(example),
            "answer": example["answer"],
        }

    rl_dataset = ds.map(map_to_rl, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        num_train_epochs=1,
        max_steps=script_args.num_train_steps,
        warmup_steps=script_args.warmup_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to=["none"],
    )

    grpo_config = GRPOConfig(
        num_generations_per_prompt=script_args.num_generations_per_prompt,
        max_new_tokens=script_args.max_gen_length,
        top_k=0,
        top_p=1.0,
        temperature=0.8,
        kl_coef=0.02,
    )

    def reward_fn(
        prompts: List[str],
        generations: List[List[str]],
        metas: List[Dict[str, Any]],
    ) -> List[List[float]]:

        all_rewards: List[List[float]] = []

        for prompt, gens, meta in zip(prompts, generations, metas):
            target_answer = meta["answer"]
            rewards_for_prompt = []
            for g in gens:
                r = dyck_reward(prompt, g, target_answer)
                rewards_for_prompt.append(r)
            all_rewards.append(rewards_for_prompt)

        return all_rewards

    def add_metadata(example):
        return {
            "prompt": example["prompt"],
            "metadata": {"answer": example["answer"]},
        }

    rl_dataset = rl_dataset.map(add_metadata, remove_columns=rl_dataset.column_names)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=rl_dataset,
        eval_dataset=None,
        reward_fn=reward_fn,
        config=grpo_config,
    )

    trainer.train()
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)


if __name__ == "__main__":
    main()
