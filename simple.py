from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig


model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", load_in_4bit=True
)

dataset = load_dataset("kye/metamath-mistal-tokenized-16384")


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    trainer_dataset=dataset,
    max_seq_length=16384,
    peft_config=peft_config,
)

trainer.train()
