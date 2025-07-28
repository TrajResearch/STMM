
import copy
import torch
from torch import nn
from transformers import   T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
import model_glo as glo
from transformers.models.t5.modeling_t5 import T5LayerNorm,T5Stack

from numpy.random import  uniform


class CustomT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.is_decoder = config.is_decoder
        self.layernorm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 调用父类的 forward 方法
        outputs = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            layer_head_mask=layer_head_mask,
            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        if not self.is_decoder:
            # 在 encoder 中，添加 global_bert_hiddenstate
            hidden_states = outputs[0]
            global_bert_hiddenstate = glo.get_value('global_bert_hiddenstate')
            assert global_bert_hiddenstate is not None , 'global_bert_hiddenstate is empty'
            hidden_states = hidden_states + global_bert_hiddenstate
            if not return_dict:
                return (hidden_states,) + outputs[1:]
            # 将元组转换为列表
            outputs = list(outputs)

            # 修改列表中的某一项，比如修改索引为2的项
            outputs[0] = self.layernorm(hidden_states)

            # 将列表转换回元组
            outputs = tuple(outputs)


        return outputs

class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        # 重写block创建逻辑，使用CustomT5Block
        self.block = nn.ModuleList([
            CustomT5Block(config, i == 0) for i in range(config.num_layers)
        ])

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CustomT5Stack(decoder_config, self.shared)

