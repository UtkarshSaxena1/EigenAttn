import torch
import torch.nn as nn
import copy
import gc

from decompose.eigen_attn_utils import decompose_opt_layer, decompose_mpt_layer, decompose_llama_layer
from models.decompose_modules import OPTEigenAttnDecoderLayer, MptBlockEigenAttn, LlamaEigenAttnDecoderLayer


def eigenattn(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    #no quantization Only low rank
    assert(('opt' in args.net.lower()) or ('mpt' in args.net.lower()) or ('llama' in args.net.lower()) or ('mistral' in args.net.lower()) or ('qwen' in args.net.lower())) # only support OPT, MPT, Llama2 model for now

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    is_mpt = False
    is_opt = False
    is_mistral = False
    if "llama" in args.net.lower() :
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = LlamaEigenAttnDecoderLayer
        
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        is_opt = True
        DecoderLayer = OPTEigenAttnDecoderLayer
        
    elif "mpt" in args.net.lower():
        is_mpt = True
        layers = model.transformer.blocks
        model.transformer.wte = model.transformer.wte.to(dev)
        
        DecoderLayer = MptBlockEigenAttn
    
    else:
        raise ValueError("Only support for opt/mpt/llama-2/llama-3.0 now")
    
    
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_mpt = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            if self.is_mpt :
                cache["position_bias"] = kwargs["position_bias"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_mpt = is_mpt

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if is_llama :
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif is_opt:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif is_mpt:
        model.transformer.wte = model.transformer.wte.cpu()
    
    else:
        raise ValueError("Only support for opt/mpt/llama-2/llama-3.0 now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    
    attention_mask = cache["attention_mask"]


    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if is_mpt:
        position_bias = cache["position_bias"]
    else:
        position_bias = None


    for i in range(len(layers)):
        layer = layers[i].to(dev)

        if is_opt:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    args.eigen_attn_params['threshold'] = 0.98
                    error = 0.0
                    num_heads = lm.model.config.num_attention_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_opt_layer(layer, inps, args, num_heads, i)
                    

                    rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                    output_hr = torch.stack([layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)])
                    while error < args.error_budget and args.eigen_attn_params['threshold']> 0.3 and rank_kq > 64 and rank_v > 64:
                        output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)])
                        error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                        args.eigen_attn_params['threshold'] -= 0.02
                        rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))


                    #error budget has been reached, revert back to the previous SVD threshold
                    args.eigen_attn_params['threshold'] += 0.04
                    rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)

                    output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)])
                    error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))

                    # del output_hr, output_lr, basis_kq, basis_v
                    del basis_kq, basis_v
                    torch.cuda.empty_cache()

                    # obtain output of model for propagation to next layer
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        elif is_mpt:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    args.eigen_attn_params['threshold'] = 0.98
                    error = 0.0
                    num_heads = lm.model.config.num_attention_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_mpt_layer(layer, inps, args, num_heads, i, attention_mask,position_ids)
                    output_hr = torch.stack([layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0] for j in range(args.nsamples)])
                    
                    while error < args.error_budget and args.eigen_attn_params['threshold']> 0.3:
                        rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)
                        error = 0
                        output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0] for j in range(args.nsamples)])
                        error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                        args.eigen_attn_params['threshold'] -= 0.02

                    args.eigen_attn_params['threshold'] += 0.04
                    rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)

                    output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0] for j in range(args.nsamples)])
                    error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))

                    # obtain output of model for propagation to next layer
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0]
                                
                    del basis_kq, basis_v, eval_kq, eval_v, output_hr, output_lr
                    torch.cuda.empty_cache()
                    
        elif is_llama:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    args.eigen_attn_params['threshold'] = 1.0
                    error = 0.0
                    num_heads = lm.model.config.num_key_value_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_llama_layer(layer, inps, args, num_heads, i, attention_mask, position_ids)

                    max_rank_kq = layer.self_attn.k_proj.weight.shape[0]
                    max_rank_v = layer.self_attn.v_proj.weight.shape[0]

                    output_hr = torch.stack([layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0] for j in range(args.nsamples)])
                    
                    rank_kq = max_rank_kq
                    rank_v = max_rank_v
                    min_thresh = 0.45
                    while error < args.error_budget and (rank_kq > min_thresh * max_rank_kq) and (rank_v > min_thresh * max_rank_v):
                        rank_kq = num_heads * ((torch.cumsum(eval_kq, dim = 0) < args.eigen_attn_params['threshold']).sum())
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config, i).to(dev)
                        qlayer = qlayer.to(dev)

                        output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0] for j in range(args.nsamples)])
                        error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                        args.eigen_attn_params['threshold'] -= 0.02

                    args.eigen_attn_params['threshold'] += 0.04
                    rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    rank_kq = num_heads * ((torch.cumsum(eval_kq, dim = 0) < args.eigen_attn_params['threshold']).sum())
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config, i).to(dev)

                    # del basis_kq, basis_v, eval_kq, eval_v, output_hr, output_lr
                    del basis_kq, basis_v, eval_kq, eval_v
                    torch.cuda.empty_cache()

                    output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0] for j in range(args.nsamples)])
                    error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))

                    # obtain output of model for propagation to next layer
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0]
        
        logger.info(f"layer {i} error:{error} threshold:{args.eigen_attn_params['threshold']} rank_kq: {rank_kq} rank_v: {rank_v} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
        
    

        qlayer.half() 

        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model