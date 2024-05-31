# Dual Attention Transformer

> **Disentangling and Integrating Relational and Sensory Information in Transformer Architectures**  
> *Awni Altabaa and John Lafferty*  
> *arxiv: https://arxiv.org/abs/2405.16727*  
> **Abstract.** The Transformer architecture processes sequences by implementing a form of neural message-passing that consists of iterative information retrieval (attention), followed by local processing (position-wise MLP). Two types of information are essential under this general computational paradigm: "sensory" information about individual objects, and "relational" information describing the relationships between objects. Standard attention naturally encodes the former, but does not explicitly encode the latter. In this paper, we present an extension of Transformers where multi-head attention is augmented with two distinct types of attention heads, each routing information of a different type. The first type is the standard attention mechanism of Transformers, which captures object-level features, while the second type is a novel attention mechanism we propose to explicitly capture relational information. The two types of attention heads each possess different inductive biases, giving the resulting architecture greater efficiency and versatility. The promise of this approach is demonstrated empirically across a range of tasks.

This repo includes an implementation of *Dual Attention* and the *Dual Attention Transformer (DAT)*. It also contains the code used to run the experiments in the paper, instructions to reproduce the experimental results, and links to detailed experimental logs.

## Summary of Paper

The Transformer architecture can be understood as an instantiation of a broader computational paradigm implementing a form of neural message-passing that iterates between two operations: 1) information retrieval (self-attention), and 2) local processing (feedforward block). To process a sequence of objects $x_1, \ldots, x_n$, this general neural message-passing paradigm has the form

$$
\begin{align*}
x_i &\gets \mathrm{Aggregate}(x_i, {\\{m_{j \to i}\\}}_{j=1}^n)\\
x_i &\gets \mathrm{Process}(x_i).
\end{align*}
$$

In the case of Transformers, the self-attention mechanism can be seen as sending messages from object $j$ to object $i$ that are encodings of the sender's features, with the message from sender $j$ to receiver $i$ given by $m_{j \to i} = \phi_v(x_j)$. These messages are then aggregated according to some selection criterion based on the receiver's features, typically given by the softmax attention scores.

We posit that there are essentially two types of information that are essential under this general computational paradigm: 1) *sensory* information describing the features and attributes of individual objects, and *relational* information about the relationships between objects. The standard attention mechanism of Transformers naturally encodes the former, but does not explicitly encode the latter.

In this paper, we propose *Relational Attention* as a novel attention mechanism which enables routing relational information between objects. We then introduce *Dual Attention*, a variant of multi-head attention combining two distinct attention mechanisms: 1) standard Self-Attention for routing sensory information, and 2) Relational Attention for routing relational information. This in turn defines an extension of the Transformer architecture with an explicit ability to reason over both types of information.

## Outline of Codebase

Here, we briefly describe the most important components of the codebase.

**Model Implementation**
- `relational_attention.py`: This module implements *Relational Attention*, an attention mechanism for routing relational information between objects.
- `symbol_retrieval.py`: This module implements different *symbol assignment mechanisms* used in *relational attention*, including *symbolic attention*, *positional symbols*, and *position-relative symbols*.
- `dual_attention.py`: This module implements *Dual Attention*, a variant of multi-head attention combining two distinct attention mechanisms: standard Self-Attention for routing sensory information and Relational Attention for routing relational information.
- `dual_attn_blocks.py`: This module implements *Dual Attention* variants of encoder and decoder Transformer blocks, which are used to build language models, seq2seq models, vision models, etc.
- `transformer_blocks.py`: This module implements standard Transformer encoder and decoder blocks, and is used as a baseline in our experiments.
- `language_models.py`: This module implements a *Dual Attention Transformer* language model (as well as a standard Transformer language model as a baseline).
- `seq2seq_models.py`: This module implements a seq2seq encoder-decoder *Dual Attention Transformer*.
- `vision_models.py`: This module implements a *Vision Dual Attention Transformer* model, in the style of a Vision Transformer (i.e., image is split up into patches and fed to an encoder).

> ℹ️ Note that our implementation of relational attention does not have the hardware-aware optimizations of modern implementations of standard attention, and is slower as a result. We suspect that a significantly faster implementation of relational attention is possible.

**Experiments**
- `experiments/relational_games`: This subdirectory includes code associated with the "Relational Games" experiments in the paper, evaluating visual relational reasoning.
- `experiments/math`: This subdirectory includes code associated with the "Mathematical Problem-Solving" experiments in the paper.
- `experiments/tiny_stories`: This subdirectory includes code associated with the Language Modeling experiments in the paper, which use the "Tiny Stories" dataset.
- `experiments/vision`: This subdirectory includes code associated with the Vision experiments in the paper, evaluating image recognition on the ImageNet dataset.

Please see the `readme.md` files in each subdirectory for instructions on reproducing the experimental results in the paper and for links to an online portal with the experimental logs.

## Usage Examples

Everything in this repo is implemented in PyTorch as `nn.Module` objects. Thus, the implemented modules are compatible with typical pytorch training code, packages like PyTorch Lightning, torchinfo, etc.

The following code demos the creation of a *Dual Attention Transformer* Language Model.

```python
from language_models import DualAttnTransformerLM

dat_lm = DualAttnTransformerLM(
    vocab_size=32_000,    # vocabulary size
    d_model=512,          # model dimension
    n_layers=6,           # number of layers
    n_heads_sa=4,         # number of self-attention heads
    n_heads_ra=4,         # number of relational attention headsd
    dff=2048,             # feedforward intermediate dimension
    dropout_rate=0.1,     # dropout rate
    activation='swiglu',  # activation function of feedforward block
    norm_first=True,      # whether to use pre-norm or post-norm
    max_block_size=1024,  # max context length
    symbol_retrieval='symbolic_attention', # type of symbol assignment mechanism
    symbol_retrieval_kwargs=dict(d_model=512, n_heads=4, n_symbols=512), # kwargs for symbol assignment mechanism
    pos_enc_type='RoPE'   # type of positional encoding to use
)

idx = torch.randint(0, 32_000, (1, 128+1))
x, y = idx[:, :-1], idx[:, 1:]
logits, loss = dat_lm(x, y)
logits # shape: (1, 128, 32000)
```

The following code demos the creation of a *Vision Dual Attention Transformer* model.

```python
from vision_models import VisionDualAttnTransformer

img_shape = (3, 224, 224)
patch_size = (16, 16)
n_patches = (img_shape[1] // patch_size[0]) * (img_shape[2] // patch_size[1])


dat_vision = VisionDualAttnTransformer(
    image_shape=img_shape,     # shape of input image
    patch_size=patch_size,     # size of patch
    num_classes=1000,          # number of classes
    d_model=512,               # model dimension
    n_layers=6,                # number of layers
    n_heads_sa=4,              # number of self-attention heads
    n_heads_ra=4,              # number of relational attention heads
    dff=2048,                  # feedforward intermediate dimension
    dropout_rate=0.1,          # dropout rate
    activation='swiglu',       # activation function of feedforward block
    norm_first=True,           # whether to use pre-norm or post-norm
    symbol_retrieval='position_relative',
    symbol_retrieval_kwargs=dict(symbol_dim=512, max_rel_pos=n_patches+1),
    ra_kwargs=dict(symmetric_rels=True, use_relative_positional_symbols=True),
    pool='cls',                # type of pooling (class token)
)

img = torch.randn(1, *img_shape)
logits = dat_vision(img)
logits.shape # shape: (1, 1000)
```

More demos are available in the `module_demo_notebooks/` subdirectory.

## Citation

If you use natbib or bibtex please use the following citation (as provided by Google Scholar).
```bibtex
@article{altabaa2024disentangling,
    title={Disentangling and Integrating Relational and Sensory Information in Transformer Architectures},
    author={Awni Altabaa and John Lafferty},
    year={2024},
    journal={arXiv preprint arXiv:2402.08856}
}
```

If you use `biblatex`, please use the following citation (as provided by arxiv).
```bibtex
@misc{altabaa2024disentangling,
    title={Disentangling and Integrating Relational and Sensory Information in Transformer Architectures},
    author={Awni Altabaa and John Lafferty},
    year={2024},
    eprint={2405.16727},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
