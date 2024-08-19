---
{{ card_data }}
---

# {{model_name}}

<!-- Provide a quick summary of what the model is/does. -->

This is a Dual-Attention Transformer Language Model, trained on the `fineweb-edu` dataset. The model is {{msize}} parameters.


## Model Details

| Size | Training Tokens| Layers | Model Dimension | Self-Attention Heads | Relational Attention Heads | Relation Dimension | Context Length |
|--|--|--|--|--|--|--|--|
| {{msize}} | {{training_tokens}} | {{n_layers}}| {{d_model}} | {{n_heads_sa}} | {{n_heads_ra}} | {{rel_dim}} | {{block_size}} |


### Model Description

- **Developed by:** Awni Altabaa, John Lafferty
- **Model type:** Decoder-only Dual Attention Transformer
- **Tokenizer:** {{tokenizer}}
- **Language(s):** English
<!-- - **License:** MIT -->
<!-- - **Contact:** awni.altabaa@yale.edu -->
- **Date:** {{date}}

### Model Sources

- **Repository:** https://github.com/Awni00/abstract_transformer
- **Paper:** [Disentangling and Integrating Relational and Sensory Information in Transformer Architectures](https://arxiv.org/abs/2405.16727)
- **Huggingface Collection:** [Dual Attention Transformer Collection](https://huggingface.co/collections/awni00/dual-attention-transformer-66c23425a545b0cefe4b9489)


## Model Usage

Use the code below to get started with the model. First, install the `dual-attention` [python package hosted on PyPI](https://pypi.org/project/dual-attention/) via `pip install dual-attention`.

To load directly from huggingface hub, use the HFHub wrapper.
```
from dual_attention.hf import DualAttnTransformerLM_HFHub

DualAttnTransformerLM_HFHub.from_pretrained('awni00/{{model_name}}')
```

Alternatively, you can download the pytorch checkpoint containing the state dict.

To download the PyTorch checkpoint, run:
```wget https://huggingface.co/awni00/{{model_name}}/resolve/main/pytorch_checkpoint.pt```

Then, you can load model weights via:
```
from dual_attention.language_models import DualAttnTransformerLM

ckpt = torch.load(ckpt_path)
model_config = ckpt['config']
model_state_dict = ckpt['model']

model = DualAttnTransformerLM(**model_config)
model.load_state_dict(model_state_dict)
```

## Training Details

The model was trained using the following setup:
- **Architecture:** Decoder-only Dual Attention Transformer 
- **Framework:** PyTorch
- **Optimizer:** AdamW
- **Learning Rate:** 6e-4 (peak)
- **Weight Decay:** 0.1
- **Batch Size:** 524,288 Tokens
- **Sequence Length:** {{block_size}} tokens
- **Total Training Tokens:** {{training_tokens}} Tokens

For more detailed training information, please refer to the paper.

## Evaluation

See paper.


## Model Interpretability Analysis

The [DAT-LM-Visualization app](https://huggingface.co/spaces/awni00/DAT-LM-Visualization/) is built to visualize the representations learned in a Dual Attention Transformer language model. It is hosted on Huggingface spaces using their free CPU resources. You can select a pre-trained DAT-LM model, enter a prompt, and visualize the internal representations in different parts of the model. You can also run the app locally (e.g., to use your own GPU) via the PyPI package.

Also, see paper.

## Citation

```
@misc{altabaa2024disentanglingintegratingrelationalsensory,
      title={Disentangling and Integrating Relational and Sensory Information in Transformer Architectures}, 
      author={Awni Altabaa and John Lafferty},
      year={2024},
      eprint={2405.16727},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.16727},
}
```