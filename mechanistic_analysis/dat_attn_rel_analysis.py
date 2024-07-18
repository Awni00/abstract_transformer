import torch
import torchinfo
from mechanistic_analysis.mechanistic_forward_old import symbolic_attn_forward_get_weights, block_forward_get_weights, datlm_forward_w_intermediate_results
import os
import sys; sys.path.append('..')
# from dual_attention_transformer import DualAttnTransformerLM
from dual_attention_transformer_old import DualAttnTransformerLM
from language_models import TransformerLM

from bertviz import head_view, model_view
import gc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tiktoken


model_name = 'DAT-ra8sa8nr32-368M'
# model_name = 'DAT-sa16ra16nr64-1.3B'
out_dir = f'analysis_results/{model_name}'
if os.path.exists(out_dir):
    print('directory exists. will overwrite files in directory, if files exist')
else:
    os.mkdir(out_dir)
    print(f'created {out_dir} directory')
print(f'saving results to {out_dir}')
model_path = '../log/DAT-ra8sa8nr32-368M_2024_07_11_18_17_07/model_10000.pt'
# model_path = '../log/DAT-ra8sa8nr32-ns512sh1-368M_2024_07_15_22_00_26/model_05000.pt'
# model_path = '../log/DAT-sa16ra16nr64-1.3B_2024_07_13_21_59_55/model_05000.pt'


enc = tiktoken.get_encoding("gpt2")

def load_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model_config = ckpt['config']

    model_state_dict = ckpt['model']
    model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}

    if 'n_heads_ra' in model_config:
        model = DualAttnTransformerLM(**model_config)
    else:
        model = TransformerLM(**model_config)

    model.load_state_dict(model_state_dict)

    return model

model_dat = load_from_ckpt(model_path).to('cuda')
model_dat = model_dat.eval()

torchinfo.summary(model_dat)

print(f'Number of parameters: {sum(p.numel() for p in model_dat.parameters() if p.requires_grad):,}')

string = """
A finite-state machine (FSM) or finite-state automaton (FSA, plural: automata), finite automaton, or simply a state machine, is a mathematical model of computation.
"""
# It is an abstract machine that can be in exactly one of a finite number of states at any given time. The FSM can change from one state to another in response to some inputs; the change from one state to another is called a transition. 
# An FSM is defined by a list of its states, its initial state, and the inputs that trigger each transition. Finite-state machines are of two types—deterministic finite-state machines and non-deterministic finite-state machines. For any non-deterministic finite-state machine, an equivalent deterministic one can be constructed.
# The behavior of state machines can be observed in many devices in modern society that perform a predetermined sequence of actions depending on a sequence of events with which they are presented. Simple examples are: vending machines, which dispense products when the proper combination of coins is deposited; elevators, whose sequence of stops is determined by the floors requested by riders; traffic lights, which change sequence when cars are waiting; combination locks, which require the input of a sequence of numbers in the proper order.

# The finite-state machine has less computational power than some other models of computation such as the Turing machine. The computational power distinction means there are computational tasks that a Turing machine can do but an FSM cannot. This is because an FSM's memory is limited by the number of states it has. A finite-state machine has the same computational power as a Turing machine that is restricted such that its head may only perform "read" operations, and always has to move from left to right. FSMs are studied in the more general field of automata theory.

tokens = torch.tensor(enc.encode(string)).unsqueeze(0).to('cuda')
tokenized_text = [enc.decode_single_token_bytes(i).decode('utf-8') for i in tokens[0]]

print('tokenized text')
print(tokenized_text)
print(f'# of tokens: {len(tokenized_text)}')

logits, intermediate_results = datlm_forward_w_intermediate_results(model_dat, tokens)

print('checking that forward call w intermediate results produces the same output')
print((model_dat(tokens)[0] - logits).abs().max()) # check that datlm_forward_w_intermediate_results produces the same result as model_dat

sa_attn_scores = [x.cpu() for x in intermediate_results['sa_attn_scores']]
html_out = head_view(sa_attn_scores, tokenized_text, html_action='return')
with open(f"{out_dir}/sa_attn_scores_head_view.html", 'w') as file:
    file.write(html_out.data)

html_out = model_view(sa_attn_scores, tokenized_text, html_action='return', display_mode="light")
with open(f"{out_dir}/sa_attn_scores_model_view.html", 'w') as file:
    file.write(html_out.data)

ra_attn_scores = [x.cpu() for x in intermediate_results['ra_attn_scores']]
html_out = model_view(ra_attn_scores, tokenized_text, html_action='return', display_mode="light")
with open(f"{out_dir}/ra_attn_scores_model_view.html", 'w') as file:
    file.write(html_out.data)

ra_rels = [rels.transpose(-1, 1).cpu() for rels in intermediate_results['ra_rels']]
html_out = head_view(ra_rels, tokenized_text, html_action='return')
with open(f"{out_dir}/ra_rels_head_view.html", 'w') as file:
    file.write(html_out.data)

html_out = model_view(ra_rels, tokenized_text, html_action='return', display_mode='light')
with open(f"{out_dir}/ra_rels_model_view.html", 'w') as file:
    file.write(html_out.data)