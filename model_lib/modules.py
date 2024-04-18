import torch
import torch.nn as nn
from typing import List, Optional
import torch.utils.checkpoint
from model_lib.moMA_generator import MoMA_generator
from transformers.activations import ACT2FN

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX

def add_function(model):
    def my_llava_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        (_,position_ids,attention_mask,_,inputs_embeds,_) = self.prepare_inputs_labels_for_multimodal(input_ids,position_ids,attention_mask,None,None,images)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs[0]
    
    model.my_llava_forward = my_llava_forward


class LlamaMLP_mapping(nn.Module):
    def __init__(self, hidden_size,hidden_size_out):
        super().__init__()
        self.hidden_size, self.hidden_size_out = hidden_size,hidden_size_out
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size_out, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.hidden_size_out, bias=False)
        self.down_proj = nn.Linear(self.hidden_size_out, self.hidden_size_out, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.act_fn_output = ACT2FN["tanh"]
        self.init_linear()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def init_linear(self):
        torch.nn.init.xavier_normal_(self.gate_proj.weight) 
        self.gate_proj.weight.data=self.gate_proj.weight.data/4.0
        torch.nn.init.xavier_normal_(self.up_proj.weight) 
        self.up_proj.weight.data=self.up_proj.weight.data/4.0
        torch.nn.init.xavier_normal_(self.down_proj.weight) 
        self.down_proj.weight.data=self.down_proj.weight.data/4.0

class MoMA_main_modal(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.moMA_generator = MoMA_generator(self.device,args)
        self.unet = self.moMA_generator.pipe.unet
        self.vae = self.moMA_generator.pipe.vae
        
        model_name = get_model_name_from_path(args.model_path)
        self.tokenizer_llava, self.model_llava, self.image_processor_llava, self.context_len_llava = load_pretrained_model(args.model_path, None, model_name, device=args.device)
        
        add_function(self.model_llava)

        self.mapping = LlamaMLP_mapping(4096,1024).to(self.device, dtype=torch.bfloat16)
        self.load_saved_components()
        self.freeze_modules()

    def load_saved_components(self):
        #load attention adapters and self cross attentions
        state_dict = torch.load(self.args.load_attn_adapters, map_location="cpu")
        self.moMA_generator.image_proj_model.load_state_dict(state_dict["projectors"])
        attn_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        attn_layers.load_state_dict(state_dict["self_cross_attentions"],strict=False)

        #load fine tuned Multi-modal LLM checkpoint
        load_params = torch.load(self.args.load_MLLM, map_location="cpu")
        self.load_state_dict(load_params,strict=False)

    def freeze_modules(self): 
        all_modules = [self.moMA_generator.pipe.vae,self.moMA_generator.pipe.text_encoder,self.unet,self.model_llava,self.mapping]
        for module in all_modules:
            module.train = False
            module.requires_grad_(False)

    def forward_MLLM(self,batch):
        llava_processeds,subjects,prompts = batch['llava_processed'].half().to(self.device),batch['label'],batch['text']
        
        input_ids,attention_masks,position_ids = [],[],[]
        for subject,prompt in zip(subjects,prompts):
            prompt_construct = f"USER: <image>\n A photo of a {subject}. Describe a new image of the same {subject} in: {prompt}. ASSISTANT: *" 
            input_id = tokenizer_image_token(prompt_construct, self.tokenizer_llava, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            attention_mask = torch.ones(input_id.shape, dtype=torch.long, device=self.device)
            position_id = torch.tensor(list(range(input_id.shape[-1])), device=self.device)
            
            position_ids += [position_id]
            attention_masks += [attention_mask[0]]
            input_ids += [input_id[0]] 
        
        input_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[-1])  for i in input_ids],batch_first=True,padding_value=self.tokenizer_llava.pad_token_id).flip(dims=[1]) 
        position_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[-1])  for i in position_ids],batch_first=True,padding_value=self.tokenizer_llava.pad_token_id).flip(dims=[1]) 
        attention_masks = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[-1])  for i in attention_masks],batch_first=True,padding_value=self.tokenizer_llava.pad_token_id).flip(dims=[1]) 
        
        output = self.model_llava.my_llava_forward(self.model_llava,input_ids=input_ids,attention_mask=attention_masks,position_ids=position_ids,images=llava_processeds)
        output = self.mapping(output)
        return output[:,-1,:]

    def reset(self):
        self.moMA_generator.reset_all()

