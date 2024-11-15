Thank you for your review and your helpful comments. We appreciate your positive feedback regarding the novelty of our proposed architecture, it being a natural extension of the Transformer framework, and the strength of the empirical results.

We hope to address each of your concerns in turn, and look forward to further discussion with you!

Below is a brief summary of our responses to the concerns you raised:

- Concern: framing and terminology around propagation of sensory and relational information in standard attention vs relational attention.
    - Response summary: we clarify the terminology here; the main point here is that we refer to the information being propagated (the values), not the attention selection criterion (attention scores), when we distinguish between sensory and relational information.
- Concern: effect of weight-tying of query/key maps in relational attention in experiments of section 4.1.
    - Response: we clarify that the attention scores are computed identically in relational attention and standard attention, and that  weight-tying only implemented in the relations (which does not apply to standard attention). We also carry out additional experiments with weight-tying $W_q^{attn} = W_k^{attn}$ in standard attention, but found no improvement.
- Concern: use of positional encoding and its relationship to symbol assignment mechanisms.
    - Response: We clarify that the same positional encoding methods is used in all models, and that positional encoding applies to the attention scores whereas symbols apply to the values.
- Concern: linguistic interpretation of the relational representations learned in the *Dual Attention Transformer* language models presented in Figure 5.
    - Response: we discuss this in more detail, and will make appropriate revisions to this section to reflect the underlying complexity.

We will address each of these in greater detail below.

## Clarification about sensory and relational information in attention

> First, the claim that standard attention mechanisms only represent sensory information is empirically false. The authors themselves cite several works describing how attention in language models often captures syntactic information, which is inherently relational.

There is a subtle but crucial distinction here, and we thank you for raising this question and the opportunity to address this. **When we say that standard attention captures sensory information while relational attention captures relational information, *we are referring to the information being propagated (i.e., the values), not the attention scores.***

In standard attention, the *attention scores* can be interpreted as relations that model a selection criterion for information retrieval. In the case of language models, these are observed to be correlated to *syntactic relations*, as mentioned in the paper. However, the key is that ***in standard attention the values retrieved are sensory*** (object embeddings), not relational. The attention score relations are computed as an intermediate step in an information retrieval operation, but the relations themselves are not explicitly represented in the updated embeddings. This observation has also been made in prior work, including [1,2].

By contrast, in our proposed *relational attention* operation, ***the values retrieved represent relations between the receiver (query object) and sender (context object).*** A key aspect of our proposal is that it decouples the relations used to model the attention scores from relations in the value embeddings.
While standard attention and relational attention model the selection mechanism (i.e., attention scores) in the same way, the values retrieved are sensory (object embeddings) in the former but relational (a separate set of learned relations) in the latter.

Thus, when we say standard attention captures sensory information while relational attention captures relational information, what we mean is that the *values* being retrieved are sensory or relational, resp. We appreciate you raising this question; we think other readers may have the same confusion. ***We will clarify this early on in the revised paper, emphasizing the distinction between the attention scores and the values with respect to where relations are represented.***

---
References

[1] Kerg, Mittal, Rolnick, Bengio, Richards, Lajoi, "Inductive biases for relational tasks", 2022

[2] Altabaa, Webb, Cohen, Lafferty, "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers", 2024

## "Abstract Relational Information" in deeper layers of ViTs

> Furthermore, recent work has explicitly found that finetuned ViTs represent sensory information in early layers, but represent abstract relational information in their later layers ...

We thank you for pointing us to these interesting references.

We would like to clarify that our claim about standard attention being sensory while relational attention being relational refers only to a **single layer of attention**.
A deep Transformer model is of course more complicated, and relational representations can emerge, for example by composing multiple layers of attention together with MLPs that learn to disentangle objects and compute relations between them.
We do not claim that it is impossible for a sufficiently-large Transformer model to learn relational representations.

Rather, **our claim is that imbuing a Transformer with explicit relational computational mechanisms (e.g., relational attention) makes it *more efficient* and *more effective***.
This is supported by our experimental results, where we see improved performance with respect to parameter efficiency and data efficiency.
<!-- Moreover, since our architecture combines sensory and relational computational mechanisms, this enables new types circuits that compose the two operations, yielding greater expressive power.  -->

## Weight-tying and symmetry of relations

> The proposed method has at least two separate important components: the representation of relational information, and the tying of key and query matrices.

First, a couple points of clarification on weight-tying of key and query matrices:
- Recall that relational attention has two sets of query/key maps: one for computing attention scores $W_q^{attn}, W_k^{attn}$ as in standard attention, and one for computing relations $W_q^{rel}, W_k^{rel}$. The weight-tying in the relational games experiments of section 4.1 refers to the *relations*, i.e., $W_q^{rel} = W_k^{rel}$, *not* the attention scores. The **attention scores are computed identically in our model and the Transformer baseline *without* weight-tying, i.e., $W_q^{attn} \neq W_k^{attn}$ in both**.
- The importance of symmetry as an inductive bias in relational learning was discussed in prior work that considered the relational games benchmark as well, e.g., [1]. The intuition is that the tasks in this benchmark rely on relatively simple same/different relations, which are inherently symmetric, which makes weight-tying a useful inductive bias.
- It's relevant to note that symmetric relations (via weight-tying $W_q^{rel} = W_k^{rel}$) only seem to be import in the relational games experiments of section 4.1. In the image recognition experiments of section 4.3, we see no significant difference between the performance of *DAT* models with symmetric or asymmetric relations. The experiments of section 4.2 and 4.4 do *not* use weight-tying.

> It appears very important to test a variant of the standard transformer subject to tied key and query matrices. For example, Figure 8 shows that removing this symmetry condition deteriorates the benefits of relational attention quite a bit, especially in low-data regimes.

As explained above, Figure 8 in the appendix explores the effects of tying the *relations'* query and key maps, not the attention scores. The attention scores are computed in the same way in all models, without weight-tying.

Nonetheless, as you suggested, one may be curious about the effect of weight-tying $W_q^{attn} = W_k^{attn}$ in the *Transformer* baseline. Although this is a distinct mechanism, with a different interpretation, it is still interesting, especially because, as you noted, weight-tying $W_q^{rel} = W_k^{rel}$ in relational attention results in significant data-efficiency improvements. Would weight-tying $W_q^{attn} = W_k^{attn}$ in standard attention result in a similar improvement?

***As suggested, we carried out additional experiments to explore this question, evaluating learning curves on a Transformer baseline with a symmetric self-attention operation.*** We found that this does *not* improve performance.

> This should also be done for the experiments presented in section 4.3.

Please see Appendix C.3, which includes an ablation over symmetry for the image recognition experiments of section 4.3. Unlike the experiments in Section 4.1, we find that symmetry of relations in relational attention does not have a significant effect, and the performance difference is within the margin of error. Our interpretation of this is that the synthetic relational games experiments of section 4.1 have a particular structure that makes symmetry a useful inductive bias (as noted in previous work as well), but more complex tasks such as image recognition may involve both symmetric and asymmetric relations.

[TODO: Elaborate a bit. Add figure to paper?]

<!-- We clarify that Figure 8 in the appendix explores tying the query/key matrices in the *relations* $r(x_i, x_j)$ of relational attention, not in the attention scores $\alpha_{ij}$. Recall that relational attention has two separate sets of query/key maps: $W_q^{rel}, W_k^{rel}$ for modeling relations and $W_q^{attn}, W_k^{attn}$ for modeling attention scores (whereas standard attention only has $W_q^{attn}, W_k^{attn}$). Here, we explore tying $W_q^{rel} = W_k^{rel}$ in relational attention, while keeping $W_q^{attn}$ and $W_k^{attn}$ as separate independent parameters, which is the same as in the Transformer baseline. The interpretation of tying $W_q^{rel} = W_k^{rel}$ is to capture *symmetric* "similarity" relations. -->

---

## Positional encoding

> Similarly, when using position-relative symbol assignment, a control that modifies the standard transformer with relative positional embeddings should also be included.

We note that ***we use the same positional encoding method in both the Transformer baselines and the DAT model in all experiments.*** Different positional encoding methods are used in different tasks (partly to match common implementations for different tasks/architectures; e.g., RoPE is used for language modeling and learned positional embeddings are used in ViT), but they are the same across different baselines within each experiment.

The symbols are separate from the positional encoding, serving a different purpose. Symbols are used only inside relational attention, whereas positional encoding is used in both standard attention and relational attention. Please see lines 268-272 for a description of how positional encoding is applied. For example, in positional encoding methods that are applied to the attention scores (e.g., RoPE), these are applied by modifying $W_q^{attn}, W_k^{attn}$, and are applied identically in standard attention and relational attention.

Recall that position-relative symbols, although related to position-relative encoding in that they encode position-relative information, are a distinct concept. For example, the position-relative bias of models like T5 modify the *attention scores* by adding a *bias*. Position-relative symbols do not touch the attention scores, and are instead part of the "values", serving as an annotation for the retrieved relations that refers or points to the source object in the relation.

It may be relevant to note that we do not test for length-generalization specifically in these experiments: the training sequences are the same length as the test sequences. We agree that length generalization is an important aspect, and that positional encoding is crucial to length generalization, but we view the design of relational architectural mechanisms as mostly orthogonal to positional encoding methods.

## Semantic vs Syntactic Relations in Attention Scores of Standard Transformers

We appreciate your engagement and attention to detail here! Thank you also for the specific reference.

We agree that much more exploration is needed to understand what types of relations the relational attention mechanism captures. The brief discussion on this in the paper reflects our initial qualitative observations, but a more through quantitative investigation is needed. We agree that the distinction is not as clear and simple as "purely syntactic" vs "purely semantic", and will revise the text in that section of the paper to emphasize the underlying complexity. We also hope to provide a deeper discussion below, relating back to the gpt2-small example you gave.

<!-- Relational attention also has attention scores for modeling selection criterion (which behave similarly to the attention scores in standard attention), but here we only consider the relations $r(x_i, x_j)$.  -->
First, we'd like to make a couple of clarifications and share our conceptual model for understanding the different types of circuits captured by relational attention and standard attention. Note that we are qualitatively comparing the relations $r(x_i, x_j)$ in relational attention to the attention scores $\alpha_{ij}$ in standard attention.
Although both $r(x_i, x_j)$ and $\alpha_{ij}$ can be described as "relations", they are used in very different ways in their respective models. Recall that the attention scores $\alpha_{ij}$ are used to model which token to attend to (i.e., a "selection criterion"), but do not explicitly enter the updated embeddings, whereas the relations $r(x_i, x_j)$ are used to directly update the receiver's embedding. Note also that $r(x_i, x_j)$ are distributed vector representations, and are *not* normalized like the attention scores $\alpha_{ij}$. Thus, $r(x_i, x_j) \in \reals^{d_r}$ is a dense representation of the relations between the two objects.
Thus, one would expect that the types of relations that would be most useful for each to be structurally different.

To understand the attention scores $\alpha_{ij}$ in standard attention, we need to think about what types of selection criteria would be useful. It is intuitively clear why syntactic relations would be useful selection criteria in the attention scores (e.g., retrieval based on subject-predicate relations). Similarly, to understand the relations in relational attention, we need to think about the types of comparisons/relations that would be useful for the given task (e.g., in language modeling, different semantic or syntactic relations) and how they may form useful computational circuits.

Some observations about the attention patterns of gpt2-small:
- Layer0Head0 appears to attend based on semantic similarity, similar to the activations of the relations in relational attention, as you point out. For example, at the `model` token, the highest attention score is to the previous token (`mathematical`), but also has high scores on `finite` and `autom`.
- Some heads in gpt2-small attend to exact copies of the same token. For example, Layer0Head5. This type of strict selectivity does not seem to occur in the relations of relational attention, which instead form of a dense distributed representation.
- Some heads in gpt2-small attend primarily based on position to the recent history. For example, Layer0Head7. Again, this would not apply to the relations in relational attention. However, we do observe that *attention scores* $\alpha_{ij}$ in relational attention *do* tend to have similar patterns.
- Attention scores $\alpha_{ij}$ often have "sinks" (e.g., to the first token, or to punctuation). The same does not apply to the distributed relations $r(x_i, x_j)$.

We have a limited understanding of what each of those different heads are doing in each model. A deeper analysis would be possible through different mechanistic interpretability tools, such as ablation of the heads.

There are many unanswered questions, and much more to explore. In the deanonymized version of the paper, we will share a link to interactive app for exploring the activations of trained *DAT* language models on different inputs. We hope this will allow people to develop intuitions about this new architecture, and facilitate follow-up work.


## Question on Figure 5

> How are you generating attention scores for tokens after “model” and “state” in Figure 5? Is this not a causal language model?

Yes, this is a causal language model. In figure 5, we are plotting the *relations* $\bm{r}_{ij} = r(x_i, x_j)$, *not the attention scores* $\alpha_{ij}$ (which would be zero for $j > i$). Recall that the same relation function $r(\cdot, \cdot)$ is applied across all pairs of objects. While the relation to future objects will be masked out by the attention scores, we can still inspect $\bm{r}_{ij}$ for the purposes of interpretability.

Although this is explained in the caption (L507), we will make sure to emphasize this and clarify that these are not attention scores to avoid the confusion. This point of confusion may be part of some of your other concerns (e.g., on weight-tying or interpretation of attention scores).

---

Thank you again for your review. Please let us know if you have any remaining concerns. We look forward to your response and continued discussion.