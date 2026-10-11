
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Qwen chat templates are based on
# - Official examples from Qwen repo: https://github.com/QwenLM/Qwen/blob/5aa84bdfd3237b37f01bc88cd49b3279b9a71d0b/examples/vllm_wrapper.py#L32
# - Online guidelines: https://github.com/Ki-Seki/chat_prompt_templates?tab=readme-ov-file#qwen-prompt-template

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T

def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer-1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer-1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer-1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        model_kwargs.update({"use_flash_attn": True})
        if dtype != "auto":
            model_kwargs.update({
                "bf16": dtype==torch.bfloat16,
                "fp16": dtype==torch.float16,
                "fp32": dtype==torch.float32,
            })

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.pad_token_id = tokenizer.eod_id # See https://github.com/QwenLM/Qwen/blob/main/FAQ.md#tokenizer

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.transformer.h

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)


def orthogonalize_qwen2_weights(model, direction):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction)
    for block in model.model.layers:
        for projection in (block.self_attn.o_proj, block.mlp.down_proj):
            projection.weight.data = get_orthogonalized_matrix(
                projection.weight.data.T, direction).T


def act_add_qwen2_weights(model, direction, coeff, layer):
    projection = model.model.layers[layer - 1].mlp.down_proj
    projection.bias = torch.nn.Parameter(
        (coeff * direction).to(projection.weight))


class Qwen2Model(QwenModel):
    """Qwen2/Qwen2.5 adapter; keep the original Qwen adapter unchanged."""

    def __init__(self, model_name_or_path, revision=None):
        self.revision = revision
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=None):
        if dtype is None:
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=self.revision, torch_dtype=dtype,
            device_map="auto", attn_implementation="eager",
        ).eval()
        if model.config.model_type != "qwen2":
            raise ValueError(f"Expected qwen2 architecture, got {model.config.model_type}")
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=self.revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_chat, tokenizer=self.tokenizer,
            system=SAMPLE_SYSTEM_PROMPT, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            QWEN_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return [self.tokenizer.encode(text, add_special_tokens=False)[0]
                for text in ("I", "As")]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction):
        return functools.partial(orthogonalize_qwen2_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction, coeff, layer):
        return functools.partial(act_add_qwen2_weights, direction=direction, coeff=coeff, layer=layer)
