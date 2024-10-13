import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from decompose.eigen_attn import eigenattn
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.modelling_mpt_eigen_attn import MptEigenAttention
from models.modelling_llama_eigen_attn import LlamaEigenAttention
from transformers.models.mpt.modeling_mpt import MptAttention
from transformers import AutoConfig
import transformers


import pdb


torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "mpt-7b"
    "mpt-30b"
    "Llama-3-8b"
    "Llama-3-80b"
]


def get_model_size(model):

    num_elements = 0

    for key in model.model.state_dict().keys():
        if 'lm_head' in key:
            continue
        param = model.model.state_dict()[key]
        num_elements += param.numel()
    
    model_size = num_elements * 16 / (8*1024*1024*1024) #size calculated in gigabytes
    return num_elements, model_size

def get_kvcache_size_opt(lm):

    num_elements = 0
    seq_len = lm.seqlen
    for key in lm.model.state_dict().keys():
        if ('k_proj.weight' in key or 'v_proj.weight' in key or 'v_proj_low.weight' in key or 'k_proj_low.weight' in key) :
            num_elements += seq_len * lm.model.state_dict()[key].shape[0]

    cache_size = num_elements * 16 / (8*1024*1024*1024) #size calculated in gigabytes
    return cache_size

def get_kvcache_size_mpt(lm, args):

    num_elements = 0
    num_heads = 32
    seq_len = lm.seqlen
    total_rankkq = 0
    total_rankv = 0
    i = 0
    for name, m in lm.model.named_modules(): 
        if ('.attn' in name and isinstance(m, MptAttention)) :
            #full rank model
            num_elements += seq_len * m.Wqkv.weight.shape[0]*2/3 #key and value layer. remove contribution from query layer
        elif (('.attn' in name and isinstance(m, MptEigenAttention))):
            #low rank model
            num_elements += seq_len * (m.rank_kq + m.rank_v) #key and value layer. remove contribution from query layer
            total_rankkq += m.rank_kq
            total_rankv += m.rank_v
            i += 1
        
    cache_size = (num_elements * 16)/ (8*1024*1024*1024) #size calculated in gigabytes

    return cache_size

def get_kvcache_size_llama(lm):

    num_elements = 0
    seq_len = lm.seqlen

    for key in lm.model.state_dict().keys():
        if ('k_proj.weight' in key or 'v_proj.weight' in key or 'v_proj_low.weight' in key or 'k_proj_low.weight' in key)  :
            num_elements += seq_len * lm.model.state_dict()[key].shape[0]
            


    cache_size = num_elements * 16 / (8*1024*1024*1024) #size calculated in gigabytes
    return cache_size


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
        elif 'mpt' in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.blocks)

            input_device = lm.model.transformer.blocks[0].device
            output_device = lm.model.transformer.blocks[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.wte.to(input_device)
            lm.model.transformer.norm_f.to(output_device)
            lm.model.lm_head.to(output_device)


    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "mpt" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)


    if args.eval_ppl:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                elif "mpt" in args.net.lower():
                    outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if args.tasks != "":
        # t_results = evaluator.simple_evaluate(
        #     lm,
        #     tasks=args.tasks,
        #     num_fewshot=args.num_fewshot,
        #     limit=None if args.limit == -1 else args.limit,
        # )
        import lm_eval
        task_manager = lm_eval.tasks.TaskManager()
        t_results = lm_eval.simple_evaluate(
            model = lm,
            tasks = args.tasks,
            num_fewshot = args.num_fewshot,
            task_manager = task_manager,
        )
        
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'mmlu' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./HF_cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile","alpaca"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--load_low_rank",  action="store_true", help="if you want to load already compressed model")
    parser.add_argument("--load_peft_model",  action="store_true", help="if you want to load the peft model")
    parser.add_argument("--peft_model_path",  default=None, help="path to finetuned model")
    parser.add_argument("--save_dir",  default=None, help="path to save the compressed and finetuned model")
    parser.add_argument("--output_dir",  default=None, help="path to save the log file")
    parser.add_argument("--evaluate_baseline",  action="store_true")
    parser.add_argument("--avg_dim", type = int)
    parser.add_argument("--error_budget", type = float, default = 0.025)
    parser.add_argument("--fine_tune", action="store_true", help="if you want to finetune model after compressing")


    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ['HF_HOME'] = args.cache_dir


    if args.net is None:
        args.net = args.model.split('/')[-1]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    print("error_budget : " ,args.error_budget)
    print(lm.model.config)


    #get baseline results
    if args.evaluate_baseline:
        logger.info("=== Evaluating baselines ===")
        model_params, _ = get_model_size(lm)
        if 'opt' in args.net.lower():
            cache_size = get_kvcache_size_opt(lm)
        elif 'mpt' in args.net.lower():
            cache_size = get_kvcache_size_mpt(lm, args)
        elif 'llama' in args.net.lower() or 'mistral' in args.net.lower() or 'qwen2' in args.net.lower():
            # cache_size = get_kvcache_size_llama(lm)
            cache_size = get_kvcache_size_llama(lm)
        logger.info(f"Baseline model parameters : {model_params/1000000000} Billion")
        logger.info(f"Baseline model KV Cache Size : {cache_size} GB for batch size of 1")
        evaluate(lm, args,logger)
        return

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")


    logger.info("=== start low rank decomposition ===")
    tick = time.time()     
    # load calibration dataset
    cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logger.info(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)    
    args.eigen_attn_params = {
            "threshold": 1.0,
            "avg_dim_features": args.avg_dim,
            "error_budget" : args.error_budget,
        }
    if not args.load_low_rank:
        eigenattn(
            lm,
            args,
            dataloader,
            logger,
        )
    else:
        print("You are already loading a low rank compressed model.")

    # now that we have a compressed model. We know the low rank dimensions for Key, Query and Value layers. We will create another model with the known dimensions (not a clean method but works :) )
    if 'opt' in args.net.lower():
        layers = lm.model.model.decoder.layers
        low_rank_config = torch.zeros(len(layers), 2)
        for i in range(len(layers)):
            low_rank_config[i][0] = layers[i].self_attn.rank_kq
            low_rank_config[i][1] = layers[i].self_attn.rank_v

        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir = args.cache_dir)
        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

        model2 = OPTForCausalLM_EigenAttn.from_pretrained(args.model, config=config, low_rank_config = low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
        layers2 = model2.model.decoder.layers
        for i in range(len(layers2)):
            layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj.weight.data
            layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj.bias.data

            layers2[i].self_attn.q_proj_low.weight.data = layers[i].self_attn.q_proj.weight.data
            layers2[i].self_attn.q_proj_low.bias.data = layers[i].self_attn.q_proj.bias.data

            layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj.weight.data
            layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj.bias.data

            # layers2[i].self_attn.v_proj_up.weight.data = layers[i].self_attn.v_proj2.weight.data
            layers2[i].self_attn.out_proj_up.weight.data = layers[i].self_attn.out_proj_up.weight.data
            layers2[i].self_attn.out_proj_up.bias.data = layers[i].self_attn.out_proj_up.bias.data
        
        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)  
            lm.tokenizer.save_pretrained(args.save_dir) 
            torch.save(low_rank_config, os.path.join(args.save_dir,'low_rank_config.pt'))
        
    
    elif 'mpt' in args.net.lower():
            layers = lm.model.transformer.blocks
            low_rank_config = torch.zeros(len(layers), 2)
            for i in range(len(layers)):
                low_rank_config[i][0] = layers[i].attn.rank_kq
                low_rank_config[i][1] = layers[i].attn.rank_v

            from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn

            config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir = args.cache_dir)
            model2 = MptForCausalLM_EigenAttn.from_pretrained(args.model, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
            layers2 = model2.transformer.blocks
            for i in range(len(layers2)):
                layers2[i].attn.Wqkv_low.weight.data = layers[i].attn.Wqkv.weight.data


                layers2[i].attn.out_proj_up.weight.data = layers[i].attn.out_proj.weight.data
            
            lm.model = model2

            if args.save_dir:
                lm.model.save_pretrained(args.save_dir)  
                lm.tokenizer.save_pretrained(args.save_dir) 
                torch.save(low_rank_config, os.path.join(args.save_dir,'low_rank_config.pt'))
    
    elif 'llama' in args.net.lower():
            layers = lm.model.model.layers
            low_rank_config = []
            for i in range(len(layers)):
                low_rank_config.append([int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)])
            
            from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

            config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir = args.cache_dir)
            model2 = LlamaForCausalLM_EigenAttn.from_pretrained(args.model, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
            layers2 = model2.model.layers
            bias = config.attention_bias
            for i in range(len(layers2)):
                #K
                layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj_low.weight.data
                if bias:
                    layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj_low.bias.data
                layers2[i].self_attn.k_proj_up.weight.data = layers[i].self_attn.k_proj_up.weight.data

                #V
                layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj_low.weight.data
                if bias:
                    layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj_low.bias.data

                #O
                layers2[i].self_attn.o_proj_up.weight.data = layers[i].self_attn.o_proj_up.weight.data
                if bias:
                    layers2[i].self_attn.o_proj_up.bias.data = layers[i].self_attn.o_proj_up.bias.data
                
            
            lm.model = model2

            if args.save_dir:
                lm.model.save_pretrained(args.save_dir)  
                lm.tokenizer.save_pretrained(args.save_dir) 
                torch.save(low_rank_config, os.path.join(args.save_dir,'low_rank_config.pt'))
    
    logger.info(time.time() - tick)
    
    logger.info("=== Evaluating compressed model ===")
    model_params, _ = get_model_size(lm)
    logger.info(f"Compressed model parameters : {model_params/1000000000} Billion")
    if 'opt' in args.net.lower():
            cache_size = get_kvcache_size_opt(lm)
    elif 'mpt' in args.net.lower():
        cache_size = get_kvcache_size_mpt(lm, args)
    elif 'llama' in args.net.lower() :
        cache_size = get_kvcache_size_llama(lm)
    logger.info(f"Compressed model KV Cache Size : {cache_size} GB for batch size of 1")
    evaluate(lm, args,logger)


    if args.fine_tune:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from time import perf_counter


        ds = load_dataset('tatsu-lab/alpaca')
        ds = ds.remove_columns(['input', 'output', 'instruction'])
        traindata = ds["train"]
        model = lm.model
        model.config.use_cache = False

        tokenizer = lm.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        lora_alpha = 64
        lora_dropout = 0.05
        lora_r = 64  # rank

        if "mpt" in args.net.lower():
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "Wqkv_low",
                    "out_proj_up",
                ],  # we will only create adapters that target for q, v metrices of attention module
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif "llama" in args.net.lower():
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "k_proj_low",
                    "k_proj_up",
                    "v_proj_low",
                    "o_proj_up",
                ],  # we will only create adapters that target for q, v metrices of attention module
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )

        output_dir = os.path.join(args.save_dir, 'peft_model')

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
   


if __name__ == "__main__":
    print(sys.argv)
    main()
