import os
from datetime import timedelta
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator, get_linear_schedule_with_warmup

from atom.configuration_llama import LlamaConfig
from atom.modeling_llama_together_yarn import LlamaForCausalLM
from atom.stable_adamw import StableAdamWUnfused

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)

def init_accelerator():
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=8,
        mixed_precision="bf16",
        log_with=None,
        kwargs_handlers=[timeout]
    )
    accelerator.init_trackers(
        project_name="yarn"
    )
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")
    return accelerator

def init_model(accelerator):
    output_dir = "/output/model"
    os.makedirs(output_dir, exist_ok=True)

    wandb.login()

    set_seed(42)

    config = LlamaConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    config.rope_scaling = {
        "type": "yarn",
        "factor": 16.0,
        "original_max_position_embeddings": 4096
    }

    config.max_position_embeddings = int(16.0 * 4096)

    model = LlamaForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.bfloat16,
        config=config
    )
    
    return model

def init_train_loader():
    dataset = load_dataset("kye/metamath-mistal-tokenized-16384", split='train')
    train_loader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=32
    )
    return train_loader

def init_optim_scheduler(model):
    optim = StableAdamWUnfused(model.parameters(), lr=2e-5)
    
    scheduler = get_linear_schedule_with_warmup(
        optim, num_training_steps=400, num_warmup_steps=20)
    
    return optim, scheduler

def finetune():
    accelerator = init_accelerator()
    model = init_model(accelerator)
    train_loader = init_train_loader()
    
    optim, scheduler = init_optim_scheduler(model)
    
    model, optim, train_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, scheduler
    )

    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)
    total_batch_size = (
        32 * accelerator.num_processes * 8
    )

    accelerator.print(f"Max train steps: 400")
    accelerator.print(f"Total batch size: {total_batch_size}")
    progress_bar = tqdm(
        range(400), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    resume_step = None
    if False:
        if False is not None or False != "":
            accelerator.print(
                f"Resuming from checkpoint {False}")
            accelerator.load_state(False)
            path = os.path.basename(False)
            training_difference = os.path.splitext(path)[0]
            resume_step = int(training_difference.replace("step_", ""))

    if False and resume_step is not None:
        train_loader = accelerator.skip_first_batches(
            train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    model.train()
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            loss = model(**batch).loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.log({"loss": loss.item()}, step=completed_steps)
                if isinstance(False, float):
                    accelerator.clip_grad_norm_(
                        model.parameters(), False)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

            if isinstance(None, int) and completed_steps > 0:
                if completed_steps % None == 0:
                    output_dir = f"step_{completed_steps}"
                    if None is not None:
                        output_dir = os.path.join(None, output_dir)
                    accelerator.save_state(output_dir)

        if completed_steps >= 400:
            break

    accelerator.print("Training Finished")
    accelerator.end_training()

    accelerator.print(f"Saving model to {output_dir}")
    if output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            f"{output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        accelerator.print("Saving Finished")

if __name__ == "__main__":
    finetune()
