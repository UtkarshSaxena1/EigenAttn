from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model
from time import perf_counter

from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
from models.modelling_mistral_eigen_attn import MistralForCausalLM_EigenAttn
import os



ds = load_dataset('tatsu-lab/alpaca')
ds = ds.remove_columns(['input', 'output', 'instruction'])
traindata = ds["train"]
breakpoint()

# model_name = "mosaicml/mpt-7b"
model_name = "mistralai/Mistral-7b-v0.1"
# model_name = 'meta-llama/Llama-2-13b-hf'
# model_name = 'meta-llama/Meta-Llama-3-8b'
cache_dir="/home/nano01/a/saxenau/EigenAttn/OmniQuant/HF_cache"
config = AutoConfig.from_pretrained(
            model_name , cache_dir = cache_dir
        )

# save_dir = '/home/min/a/saxenau/NANO/EigenAttn/OmniQuant/Meta-Llama-3-8b/avgdim_1_nsamples_128_errorbudget_0.09/saved_model/'
# save_dir = '/home/min/a/saxenau/NANO/EigenAttn/OmniQuant/mpt-7b/avgdim_128_nsamples_128_errorbudget_0.008/saved_model/'
save_dir = '/home/min/a/saxenau/NANO/EigenAttn/OmniQuant/Mistral-7B-v0.1/avgdim_1_nsamples_512_errorbudget_0.0/saved_model/'
# save_dir = '/home/min/a/saxenau/NANO/EigenAttn/OmniQuant/Llama-2-7b-hf/avgdim_1_nsamples_128_errorbudget_0.09/saved_model/'
# save_dir = '/home/min/a/saxenau/NANO/EigenAttn/OmniQuant/Llama-2-13b-hf/avgdim_1_nsamples_128_errorbudget_0.07/saved_model/'
low_rank_config = torch.load(os.path.join(save_dir,'low_rank_config.pt'))

# model = LlamaForCausalLM_EigenAttn.from_pretrained(save_dir, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=cache_dir)
# model = MptForCausalLM_EigenAttn.from_pretrained(save_dir, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=cache_dir)
model = MistralForCausalLM_EigenAttn.from_pretrained(save_dir, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=cache_dir)

# if exists use config on cache
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,legacy=False, cache_dir = cache_dir, trust_remote_code=True, return_token_type_ids=False)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



# Parameter efficient finetuning for LoRA configuration

# ####### peft config for Llama-3 #####
# lora_alpha = 64
# lora_dropout = 0.05
# lora_r = 64  # rank

# peft_config = LoraConfig(
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     target_modules=[
#         "k_proj_low",
#         "k_proj_up",
#         "v_proj_low",
#         "o_proj_up",
#     ],  # we will only create adapters that target for q, v metrices of attention module
#     r=lora_r,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

####### peft config for Mistral #####
lora_alpha = 64
lora_dropout = 0.05
lora_r = 64  # rank

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=[
        "k_proj_low",
        "k_proj_up",
        "v_proj_low",
        "v_proj_up",
    ],  # we will only create adapters that target for q, v metrices of attention module
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

####### peft config for Mpt #####
# lora_alpha = 64
# lora_dropout = 0.05
# lora_r = 64  # rank

# peft_config = LoraConfig(
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     target_modules=[
#         "Wqkv_low",
#         "out_proj_up",
#     ],  # we will only create adapters that target for q, v metrices of attention module
#     r=lora_r,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
import transformers

output_dir = "rebuttal-mistral-7b-v0.1-0.6x"
training_arguments = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,
    num_train_epochs=1,
    max_steps=2000,
    fp16=True,
    push_to_hub=False,
)

# creating trainer with the training agruments
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=traindata,
    peft_config=peft_config,  # passing peft config
    dataset_text_field="text",  # mentioned the required column
    args=training_arguments,  # training agruments
    tokenizer=tokenizer,  # tokenizer
    packing=False,
    max_seq_length=2048,
)

start_time = perf_counter()
trainer.train()
end_time = perf_counter()
training_time = end_time - start_time
print(f"Time taken for training: {training_time} seconds")


# from peft import PeftModel
# ft_model = PeftModel.from_pretrained(model, "/home/min/a/saxenau/NANO/EigenAttn/OmniQuant/mpt_qlora_finetuned_7b_hf/checkpoint-500/")
# print(ft_model)

