import torch
import torchinfo
import os
import sys; sys.path.append('..')

from language_models import TransformerLM
from bertviz import head_view, model_view
import gc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tiktoken

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--model_dir', type=str, default='../experiments/fineweb/log', help='directory where models are stored')
parser.add_argument('--out_dir', type=str, default='analysis_results', help='output directory')
parser.add_argument('--model_name', type=str, default=None, help='model name')
parser.add_argument('--old', default=0, type=int, help='use old model')
parser.add_argument('--prompt_text', type=str, default=None, help='input text')
args = parser.parse_args()

if args.old:
    from dual_attention_transformer_old import DualAttnTransformerLM
    from mechanistic_analysis.mechanistic_forward_old import symbolic_attn_forward_get_weights, block_forward_get_weights, datlm_forward_w_intermediate_results
else:
    from dual_attention_transformer import DualAttnTransformerLM
    from mechanistic_analysis.mechanistic_forward import symbolic_attn_forward_get_weights, block_forward_get_weights, datlm_forward_w_intermediate_results

# model_name = 'DAT-sa16ra16nr64-1.3B'

if os.path.exists(args.out_dir):
    print('directory exists. will overwrite files in directory, if files exist')
else:
    os.mkdir(args.out_dir)
    print(f'created {args.out_dir} directory')
print(f'saving results to {args.out_dir}')
model_path = f'{args.model_dir}/{args.model_path}'
# model_path = '../log/DAT-ra8sa8nr32-368M_2024_07_11_18_17_07/model_10000.pt'
# model_path = '../log/DAT-ra8sa8nr32-ns512sh1-368M_2024_07_15_22_00_26/model_05000.pt'
# model_path = '../log/DAT-sa16ra16nr64-1.3B_2024_07_13_21_59_55/model_05000.pt'

model_name = args.model_name if args.model_name is not None else args.model_path.split('_2024')[0]
out_dir = f'{args.out_dir}/{model_name}'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f'created {out_dir} directory')
else:
    print(f'{out_dir} directory exists. will overwrite files in directory, if files exist')

print('='*100)
print(f'Loading model from {model_path}')
print(f'Will save results to {out_dir}')
print('='*100)

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

model = load_from_ckpt(model_path).to('cuda')
model = model.eval()

is_dat = hasattr(model, 'n_heads_ra')
if not is_dat:
    raise NotImplementedError("Haven't implemented analysis for standard Transformer yet...")

torchinfo.summary(model)

print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

if args.prompt_text is not None:
    string = args.prompt_text
else:
    string = """A finite-state machine (FSM) or finite-state automaton (FSA, plural: automata), finite automaton, or simply a state machine, is a mathematical model of computation. It is an abstract machine that can be in exactly one of a finite number of states at any given time. The FSM can change from one state to another in response to some inputs; the change from one state to another is called a transition."""
# It is an abstract machine that can be in exactly one of a finite number of states at any given time. The FSM can change from one state to another in response to some inputs; the change from one state to another is called a transition. 
# An FSM is defined by a list of its states, its initial state, and the inputs that trigger each transition. Finite-state machines are of two types—deterministic finite-state machines and non-deterministic finite-state machines. For any non-deterministic finite-state machine, an equivalent deterministic one can be constructed.
# The behavior of state machines can be observed in many devices in modern society that perform a predetermined sequence of actions depending on a sequence of events with which they are presented. Simple examples are: vending machines, which dispense products when the proper combination of coins is deposited; elevators, whose sequence of stops is determined by the floors requested by riders; traffic lights, which change sequence when cars are waiting; combination locks, which require the input of a sequence of numbers in the proper order.

# The finite-state machine has less computational power than some other models of computation such as the Turing machine. The computational power distinction means there are computational tasks that a Turing machine can do but an FSM cannot. This is because an FSM's memory is limited by the number of states it has. A finite-state machine has the same computational power as a Turing machine that is restricted such that its head may only perform "read" operations, and always has to move from left to right. FSMs are studied in the more general field of automata theory.

tokens = torch.tensor(enc.encode(string)).unsqueeze(0).to('cuda')
tokenized_text = [enc.decode_single_token_bytes(i).decode('utf-8') for i in tokens[0]]

print('Input text:')
print(string)

print('Tokenized text:')
print(tokenized_text)
print(f'# of tokens: {len(tokenized_text)}')

logits, intermediate_results = datlm_forward_w_intermediate_results(model, tokens)

print('checking that forward call w intermediate results produces the same output')
print((model(tokens)[0] - logits).abs().max()) # check that datlm_forward_w_intermediate_results produces the same result as model_dat

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

print('='*50 + 'DONE' + '='*50)