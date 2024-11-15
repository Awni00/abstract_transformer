Thank you for your review. We hope to address each point raised in turn. The following is a brief summary:

- Concern: relational attention is very similar graph attention networks (GAT).
    - Response: This is inaccurate. While GAT uses a self-attention-like mechanism in its aggregation of node features, this is entirely distinct from relational attention. We explain this in detail below. You mention a couple other papers which are also unrelated, despite having some keywords in common.
- Concern: experiments are performed on a small set of simpler tasks.
    - Response: We respectfully disagree. While our experiments include synthetic benchmarks to enable controlled evaluations with respect to previously-studied relational tasks, they also include complex real-world tasks such as image recognition and language modeling. Our experiments cover a diverse set of task paradigms (sequence classification, sequence-to-sequence, autoregressive next-token prediction), data modalities (text and vision), and architectural variants (encoder-only, decoder-only, encoder-decoder, ViT-style).

## Difference between relational attention and GAT

> The relational attention sounds very similar to the graph attention network to me.

***This is inaccurate.*** The Graph attention network (GAT) layer is analogous to self-attention in standard Transformers, but it is very different from the relational attention mechanism proposed in our work. ***The only common feature between GAT and our proposed architecture is the use of an attention operation.*** We hope to provide some clarification and discussion below, highlighting the differences between GAT, Transformer attention, and our relational attention.

<!-- We are somewhat confused about why you believe relational attention is very similar to graph attention networks, and would appreciate further clarification so that we may better address your concerns. -->


<!-- While Transformer models can be interpreted as a type of graph neural network operating over a fully-connected graph, and this interpretation can be conceptually useful sometimes, Transformers are a distinct architecture with several distinguishing features. Graph attention networks (GATs) bear closer resemblance to Transformers, because they incorporate an attention mechanism (note that GATs were proposed after Transformers). -->

The standard attention mechanism of Transformers takes the form (Vaswani et al. 2017):
$$h_i' = \sum_{j} \alpha_{ij} W_v h_j,$$
where $\alpha_{ij}$ are attention scores, and $h_i$ are the hidden embeddings.

A GAT layer updates node embeddings at each layer via a similar operation (Velickovic et al. 2018):
$$h_i' = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j),$$
where $\alpha_{ij}$ are attention scores computed similarly to the dot-product attention mechanism used in Transformers. $\sigma$ is an optional non-linearity. Note that in GATs, a graph is given as input and $\mathcal{N}_i$ node $i$'s neighbors on the graph. Thus, GAT can be interpreted as applying an input-dependent weight to the aggregation operation in GNNs.

The *relational attention* mechanism proposed in our work is very different to both GAT and standard attention:
$$h_i' = \sum_{j} \alpha_{ij} (W_r r(h_i, h_j) + W_s s_j),$$
where $r(\cdot, \cdot) \in \mathbb{R}^{d_r}$ is a learned relation function, $s_j \in \mathbb{R}^{d}$ is a "symbol vector" which "points to" object $j$, and $W_r, W_s$ are learned linear maps.

(Note that we presented the single-head version for each of the three mechanisms for clarity and simplicity, but all have multi-head variants.)

As you can see, while graph attention networks have a similar form to standard Transformer attention, our proposal of relational attention bears little resemblance to graph attention networks. In particular, standard self-attention and GAT both only model a selection criterion that determines how to aggregate the neighbors' embeddings. In attention terminology, the **values in standard self-attention and GAT are the *feature embeddings of the neighbors***. By contrast, **in relational attention, the values are representations of the *relations* between the receiver (query object) and sender (context object).** This is a fundamental difference, and is the key to the proposed architecture.

## Related work

We would like to mention that our work is most influenced by a line of work on relational architectures (outside the GNN literature), including: RelationNet (Santoro et al), PrediNet (Shanahan et al), and Abstractor (Altabaa et al.).
While the graph attention network is very different from our work, there exists other work in the GNN literature that bears a closer resemblance. ***We will add an expanded related work section which discusses in-detail the relation to various existing work.***


<!-- ---

**Idea of a perceptual module followed by relational reasoning networks**
> Also the idea of having a perception module followed by relational reasoning networks have been explored for many years in visual reasoning domain (e.g. [1,2]).

--- -->

## Relation to Link Attention in the Retriever
> The Symbolic attention also sounds like Link mechanism in the Retriever[3].

***Symbolic attention and the Link mechanism of the Retriever are distinct mechanisms.*** Though the Retriever is an interesting work, it has ***different motivations and entirely different implementations***.

The Link attention mechanism in the Retriever is based on a particular definition of "content" and "style". Where, given an input sequence $X = [x_1, ..., x_n]$, the style of $X$ is defined as the permutation-invariant information, and "content" is the rest of the information. The style of $X$ is extracted by applying a permutation-invariant function to $X$, and the content extracted by a non-permutation-invariant encoder. The link attention mechanism is a cross-attention operation with queries being the extracted content tokens and the values are the extracted style tokens.

The motivation of symbolic attention is different, with *no connection to any notion of permutation-invariance*. Instead, symbolic attention is interpreted as implementing a (differentiable) equivalence class over feature embeddings, by comparing the feature embeddings to a set of learnable feature templates and retrieving an associated learnable value vector. (Note that both the keys and values are learnable parameters here.)

We agree, however, that the Retriever is an interesting work tackling related problems. We will discuss it in the expanded related work section, together with other prior work on perception and reasoning.

---

## On the experiments

> The experiments are only performed on a small set of simpler tasks. I wonder how the proposed method will perform for more complex tasks.

We respectfully disagree with this characterization. Our suite of experiments cover a range of tasks, data modalities, and architectural variants, which include both controlled synthetic tasks and large-scale complex real-world tasks. This was a recognized point of strength in all other reviews (mxrQ, YUpf, qVFZ).

Below, we aim to summarize the experimental component of the paper.

1. Sec 4.1: We begin with a synthetic benchmark of relational tasks, called "relational games". This benchmark was studied in a series of prior works on relational architectures, and gives us a way to evaluate our proposed model in a controlled environment. The benchmark contains a suite of 5 different tasks, where we evaluate learning curves (i.e., data-efficiency) and compare to standard Transformers. We show that our model is dramatically more data-efficient.
2. Sec 4.2: We evaluate **symbolic reasoning** via a set of **mathematical problem-solving** tasks. These tasks are modeled as **sequence-to-sequence** tasks, using an **encoder-decoder architecture**. We demonstrate improved performance compared to a standard Transformer, across different model sizes and parameter scales.
3. Sec 4.3: We evaluate our model on **visual processing** via **image recognition** tasks. We use a **ViT-style architecture** on these tasks, processing the input image as a sequence of patches. We demonstrate improved performance, showing that relational processing can be useful for visual processing tasks such as image recognition.
4. Sec 4.4: We evaluate our model on **autoregressive language-modeling** using a causal decoder-only architecture. We evaluate **scaling laws** with respect to both data size and model size, and show **improvements in both data efficiency and parameter efficiency compared to standard Transformers.** Our models go up to **1.3 Billon parameters,** roughly matching the scale of GPT2.

These experiments show that the relational attention mechanism, and the enhanced relational processing capabilities it enables, yield improved performance across a wide range of tasks (symbolic reasoning, image recognition, language modeling), data modalities (e.g., text, vision), and architectural variants (e.g., encoder-only, decoder-only, encoder-decoder, and ViT-style).

These experimental contributions are especially significant to the literature on relational architectures and inductive biases, as they demonstrate that relational computational mechanisms can yield performance improvements 

---

**Summary of contributions**

- [Emphasize significance of our complex real-world experiments with respect to literature on relational architectures. (i.e., [1,2,3,4] show performance benefits only on synthetic relational tasks (like the relational games benchmark in section 4.1), but not more general )]
- [Significance of relational computational mechanisms in Transformers, specifically.]
- [An extension of the Transformer framework. *DAT* can be applied to any task that a Transformer can be applied to, using any architectural variant of Transformers (e.g., Encoder-Decoder, Encoder-only, Decoder-only, ViT, etc.)]
...