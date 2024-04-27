"""Train LLaMa3 on Arazn Dataset"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
train_dataset_path = "train.jsonl"
test_dataset_path = "test.jsonl"
hf_token = "<add-your-huggingface-token>"

# QLoRA parameters
lora_alpha = 64
lora_r = lora_alpha * 2
lora_dropout = 0

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 400
logging_steps = 50

# SFT parameters
max_seq_length = 512
packing = False
device_map = "auto"

new_model = f"llama3-arazn-ar-v1"
output_dir = new_model


input_field = "code_switched"
output_field = "arabic"


raw_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Translate the following code-switched Arabic-English-mixed text to Arabic only.<|eot_id|><|start_header_id|>user<|end_header_id|>

{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


with open(train_dataset_path, "r") as f:
    lines = f.readlines()

pop_list = [1882, 1967, 2033, 2070, 2071, 2072, 2073, 2074]
for i in pop_list:
    lines.pop(i)

    
with open(train_dataset_path, "w") as f:
    f.writelines(lines)
    

train_dataset = load_dataset('json', data_files=train_dataset_path, split="train")
test_dataset = load_dataset('json', data_files=test_dataset_path, split="train")

mapper_fn = lambda examples: {
    'text': [raw_prompt.format(source=source) + target + "<|eot_id|>" 
             for source, target, in zip(examples[input_field], examples[output_field])]
}

train_dataset_mapped = train_dataset.map(mapper_fn, batched=True)
test_dataset_mapped = test_dataset.map(mapper_fn, batched=True)

print(train_dataset_mapped[0]['text'])

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

peft_config = LoraConfig(
    use_dora=True,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    token=hf_token,
    use_cache=False,
)

model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

response_template = "assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    data_collator=collator,
)

trainer.train()