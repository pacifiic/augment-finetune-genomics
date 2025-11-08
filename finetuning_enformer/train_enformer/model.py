# -*- coding: utf-8 -*-
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from enformer_pytorch_for_lora.modeling_enformer import from_pretrained
from enformer_pytorch_for_lora.finetune import HeadAdapterWrapper_For_Lymphocyte_Vector_Input
from .utils import freeze_stem_and_conv_tower, EnformerPEFTWrapper


def build_dual_enformer(use_lora: bool = True, lora_rank: int = 32,
                        target_modules=("to_q", "to_k", "to_v", "to_out")) -> nn.Module:
    if use_lora:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.1,
            bias="none",
            target_modules=list(target_modules),
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    enformer1 = from_pretrained('EleutherAI/enformer-official-rough')
    enformer2 = from_pretrained('EleutherAI/enformer-official-rough')
    freeze_stem_and_conv_tower(enformer1)
    freeze_stem_and_conv_tower(enformer2)
    if use_lora:
        wrapped1 = EnformerPEFTWrapper(enformer1)
        wrapped2 = EnformerPEFTWrapper(enformer2)
        peft1 = get_peft_model(wrapped1, lora_config)
        peft2 = get_peft_model(wrapped2, lora_config)
        model = HeadAdapterWrapper_For_Lymphocyte_Vector_Input(enformer1=peft1, enformer2=peft2)
    else:
        model = HeadAdapterWrapper_For_Lymphocyte_Vector_Input(enformer1=enformer1, enformer2=enformer2)
    return model