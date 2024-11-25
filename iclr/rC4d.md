Thank you for your review. We aim to address each point raised in detail.

Below, we provide a summary of our responses:

- Concern: relational attention is "very similar" to graph attention networks (GAT).
    - Response: This characterization is inaccurate, relational attention is a distinct mechanism from GAT. We provide a detailed explanation of the differences below.
- Concern: experiments are performed on a small set of simpler tasks.
    - Response: We respectfully disagree. While our experiments include synthetic benchmarks to enable controlled evaluations with respect to previously-studied relational tasks, they also include complex real-world tasks such as image recognition and language modeling. Our experiments span a diverse range of task paradigms (sequence classification, sequence-to-sequence, autoregressive next-token prediction), data modalities (text and vision), and architectural variants (encoder-only, decoder-only, encoder-decoder, ViT-style). Our models go up to 1.3B parameters in size, and we include an analysis of scaling laws compared to standard Transformers.

## Difference between relational attention and GAT

> The relational attention sounds very similar to the graph attention network to me.

***This characterization is inaccurate.***  The Graph attention network (GAT) layer is essentially self-attention with a mask corresponding to graph neighborhoods. Thus, it is no more similar to relational attention than standard self-attention is. ***The only common feature between GAT and our proposed relational attention mechanism is that it involves computing attention scores*** (which it shares with standard attention). We'd like to explain in more detail below.

The standard attention mechanism of Transformers (Vaswani et al. 2017) takes the form:
$$h_i' = \sum_{j} \alpha_{ij} W_v h_j,$$
where $\alpha_{ij}$ are attention scores, and $h_i$ are the hidden embeddings.

A GAT layer (Velickovic et al. 2018; Eq 4) updates node embeddings at each layer via a similar operation:
$$h_i' = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j),$$
where $\alpha_{ij}$ are attention scores computed similarly to the dot-product attention mechanism used in Transformers. $\sigma$ is an optional non-linearity. The main difference is the attentional mask representing the graph neighborhoods $\mathcal{N}_i$.

The *relational attention* mechanism proposed in our work is very different to both GAT and standard attention:
$$h_i' = \sum_{j} \alpha_{ij} (W_r r(h_i, h_j) + W_s s_j),$$
where $r(\cdot, \cdot) \in \mathbb{R}^{d_r}$ is a learned relation function, $s_j \in \mathbb{R}^{d}$ is a "symbol vector" which "points to" object $j$, and $W_r, W_s$ are learned linear maps. Instead of attending to the embeddings $h_j$ of the objects in the context, **relational attention attends to and retrieves learned *relations* $r(h_i, h_j)$ between the query object and the context objects**. Here, $r(\cdot, \cdot)$ is modeled as a series of inner product comparisons under different feature projections.

(Note that we presented the single-head version for each of the three mechanisms for clarity and simplicity, but all have multi-head variants.)

As you can see, while graph attention networks have a similar form to standard Transformer attention, our proposal of relational attention bears little resemblance to graph attention networks. In particular, standard self-attention and GAT both only model a selection criterion that determines how to aggregate the neighbors' embeddings. In attention terminology, the **values in standard self-attention and GAT are the *feature embeddings of the neighbors***. By contrast, **in relational attention, the values are representations of the *relations* between the receiver (query object) and sender (context object).** This is a fundamental difference, and is the key to our proposed architecture.

<!-- ## Related work -->
<!--  -->
<!-- We would like to mention that our work is most influenced by a line of work on relational architectures (outside the GNN literature), including: RelationNet (Santoro et al), PrediNet (Shanahan et al), and Abstractor (Altabaa et al.). ***We will add an expanded related work section which discusses in-detail the relation to various existing work.*** -->
<!-- While the graph attention network is very different from our work, there exists other work in the GNN literature that bears a closer resemblance.  -->


<!-- ---

**Idea of a perceptual module followed by relational reasoning networks**
> Also the idea of having a perception module followed by relational reasoning networks have been explored for many years in visual reasoning domain (e.g. [1,2]).

--- -->

## Relation to Link Attention in the Retriever
> The Symbolic attention also sounds like Link mechanism in the Retriever[3].

***Symbolic attention and the Link mechanism of the Retriever are entirely distinct mechanisms.*** While the Retriever is an interesting work, it has different motivations and implementations. The key idea behind the Retriever is to separate permutation-invariant information from the rest of the information. This is achieved through an autoencoder architecture, with permutation-invariant encoder. We don't see a direct connection to symbolic attention, aside from the involvement of an attention operation in the Retriever.

<!-- The Link attention mechanism in the Retriever is based on a particular definition of "content" and "style". Where, given an input sequence $X = [x_1, ..., x_n]$, the style of $X$ is defined as the permutation-invariant information, and "content" is the rest of the information. The style of $X$ is extracted by applying a permutation-invariant function to $X$, and the content extracted by a non-permutation-invariant encoder. The link attention mechanism is a cross-attention operation with queries being the extracted content tokens and the values are the extracted style tokens.

The motivation of symbolic attention is different, with *no connection to any notion of permutation-invariance*. Instead, symbolic attention is interpreted as implementing a (differentiable) equivalence class over feature embeddings, by comparing the feature embeddings to a set of learnable feature templates and retrieving an associated learnable value vector. (Note that both the keys and values are learnable parameters here.) -->


## On the experiments

> The experiments are only performed on a small set of simpler tasks. I wonder how the proposed method will perform for more complex tasks.

We respectfully disagree with this characterization. Our suite of experiments cover a range of tasks, data modalities, and architectural variants, which include both controlled synthetic tasks and large-scale complex real-world tasks. This was a recognized point of strength in all other reviews (mxrQ, YUpf, qVFZ).

Below, we aim to summarize the experimental component of the paper.

1. Sec 4.1: We begin with a synthetic benchmark of relational tasks, called "relational games". This benchmark was studied in a series of prior works on relational architectures, and gives us a way to evaluate our proposed model in a controlled environment. The benchmark contains a suite of 5 different tasks, where we evaluate learning curves (i.e., data-efficiency) and compare to standard Transformers. We show that our model is significantly more data-efficient.
2. Sec 4.2: We evaluate **symbolic reasoning** via a set of **mathematical problem-solving** tasks. These tasks are modeled as **sequence-to-sequence** tasks, using an **encoder-decoder architecture**. We demonstrate improved performance compared to a standard Transformer, across different model sizes and parameter scales.
3. Sec 4.3: We evaluate our model on **visual processing** via **image recognition** tasks. We use a **ViT-style architecture** on these tasks, processing the input image as a sequence of patches. We demonstrate improved performance, showing that relational processing can be useful for visual processing tasks such as image recognition.
4. Sec 4.4: We evaluate our model on **autoregressive language-modeling** using a causal decoder-only architecture. We evaluate **scaling laws** with respect to both data size and model size, and show **improvements in both data efficiency and parameter efficiency compared to standard Transformers.** Our models go up to **1.3 Billon parameters,** roughly matching the scale of GPT2.

These experiments show that the *DAT* architecture yields improved performance across a wide range of tasks (symbolic reasoning, image recognition, language modeling), data modalities (e.g., text, vision), and architectural variants (e.g., encoder-only, decoder-only, encoder-decoder, and ViT-style).

Moreover, we note that we build on a line of work on relational architectures and inductive biases [Ref 15-22]. The empirical evaluation of this prior work was mainly limited to synthetic tasks, like the relational games benchmark of Sec 4.1. Thus, one of the key contributions of our work is to integrate relational neural mechanisms into a general architectural framework (namely, the Transformer) and demonstrate that relational neural mechanisms and inductive biases confer performance benefits on **complex real-world tasks, like language modeling and image recognition**. We believe this is an important contribution to this line of work.

---

Thank you for your review. We hope we were able to clarify the novelty of our architectural proposals. Please let us know if we have addressed your concerns or if you have any remaining concerns. We look forward to your response.