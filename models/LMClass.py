import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn
from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
import os
from typing import List, Optional, Tuple, Union
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from peft import PeftModel


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()
        self._rank = 0
        self._world_size = 1
        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model

        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation, cache_dir = args.cache_dir
        )
        config.use_cache = False
        use_fast = False
        if 'mpt' in args.net.lower():
            use_fast = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast,legacy=False, cache_dir = args.cache_dir)
        if args.load_low_rank :
            low_rank_config = torch.load(os.path.join(args.save_dir,'low_rank_config.pt'))
            if 'mpt' in args.net.lower():
                self.model = MptForCausalLM_EigenAttn.from_pretrained(args.save_dir, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
                if args.load_peft_model:
                    self.model = PeftModel.from_pretrained(self.model, args.peft_model_path)
                    
            elif 'opt' in args.net.lower():
                self.model = OPTForCausalLM_EigenAttn.from_pretrained(args.save_dir, config=config, low_rank_config = low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
                if args.load_peft_model:
                    self.model = PeftModel.from_pretrained(self.model, args.peft_model_path)

            elif 'llama' in args.net.lower():
                self.model = LlamaForCausalLM_EigenAttn.from_pretrained(args.save_dir, config=config, low_rank_config=low_rank_config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
                if args.load_peft_model:
                    self.model = PeftModel.from_pretrained(self.model, args.peft_model_path)
                    self.model = self.model.model
            else:
                raise NotImplementedError
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16, cache_dir=args.cache_dir)
        if 'mpt' in args.net.lower():
            self.seqlen = self.model.config.max_seq_len
        else:
            self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            
            return self.model.config.max_position_embeddings
            # return self.model.config.max_seq_len

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size


    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    # def tok_encode_batch(self, strings):
    #     return self.tokenizer(
    #         strings,
    #         padding=True,
    #         add_special_tokens=False,
    #         return_tensors="pt",
    #     )

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        # add_special_tokens = {}
        # add_special_tokens = {self.tokenizer.bos_token}
        self.tokenizer.pad_token = self.tokenizer.bos_token
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            # padding_side="left",
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]
    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

