# from transformers import Trainer
import torch
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, PretrainedBartModel
from torch import  nn
import logging

logger = logging.getLogger(__name__)

class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        logger.info('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, 'use_deep'):
            self.use_deep = (config.use_deep == 'yes')
        else:
            self.use_deep = False

        deep_param = self.use_deep
        
        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None

        if self.task_mode == 'dataless':
            self.mode_para = 1
        elif self.task_mode == 'data2text' or self.task_mode == 'triples' or self.task_mode == 'webnlg' or \
                self.task_mode == 'writingPrompts':
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4

        if self.optim_prefix:
            self.mode_para = 0
            # print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            logger.info('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))

            # elif not deep_param:
            if not deep_param:
                # print('UNDER PARAMETRIZATION 1')
                # done since "directly updating prefix parameters leads to unstable optimization" (p. 4586)
                # after training, "reparametrization parameters can be dropped, and only the prefix needs to be saved"

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            logger.info(name + ' ' + str(param.shape))
            total_param += param.numel()
        logger.info('total param is {}'.format(total_param))

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                 },
                        }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)

        return result

    def forward(self,
        input_ids=None,
        gpt2_model=None,
        past_key_values=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # labels=None,
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)

        output = gpt2_model(input_ids=input_ids,
                            past_key_values=past_key_values, **kwargs)

        return output
