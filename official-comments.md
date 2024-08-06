
---

**Additional responses to comments of oNEA review**

> In Eq.3, does SymbolRetriever mean Symbol Assignment Mechanisms?

Yes. The SymbolRetriever returns a sequence of "symbols" $(s_1, ..., s_n)$, via one of the symbol assignment mechanisms: positional symbols, position-relative symbols, or symbolic attention (though in the case of the latter the computation is of course implemented differently than the first two). We will make this clear in the revised text.

>In fig.6 match pattern and fig.12 match pattern, Dual-Attn Transformer (nsa = 1, nra = 1) performs differently, why?

Fig 6 explores the effect of a symmetry inductive bias in the relations inside relational attention on the relational games benchmark. That is, it compares the learning curves of matching dual-attention Transformer models, one with a symmetry constraint on the relations in RA ($W_q^{rel} = W_k^{rel}$) and the other with no symmetry constraint. Figure 12 evaluates the performance of a Dual Attention Transformer variant with the Abstractor's RCA instead of relational attention (proposed in this work). In Figure 12, neither model has a symmetry constraint. Thus, the solid lines in Figure 12 match the dashed lines in Figure 6, since the models in Fig 12 have no symmetry constraint. We will add another ablation figure which compares relational attention against the Abstractor's RCA *with* a symmetry constraint on the relations (in the case of the Abstracor where the relations and the attention scores are the same thing, we will take 'symmetry' to mean symmetry over the attention scores).


---
**Additional responses to comments of qSAC review**


**Questions.**
> Why relational-only attention in Figure 2 works best? Any intuitions for that? Aren’t the two types of attention crucial (according to the intro)? Why can we get away with only relational attention?

This is an important question. Yes, there is a clear intuitive reason for this. The relational games tasks are "purely relational" tasks in the sense that each task is entirely solvable with only the pairwise same/different relations. Relational attention can capture this information more efficiently than standard (sensory) attention. This is also why the standard Transformer is less efficient on this task. In fact, based on prior work, what is interesting about the result is actually that the configuration with both types of attention heads is *not* significantly worse than the configuration using only relational heads. 

A common idea in previous work is that relational learning requires strict inductive biases that "filter out" non-relational information because it would corrupt the relational information and require large amounts of data to overcome. This result demonstrates that merely offering the model different computational mechanisms, including relational computational mechanisms, is sufficient and that the model can learn to use the appropriate mechanism based on the task and context. This is important because it suggests the possibility of equipping a general modeling framework with enhanced relational processing capabilities without sacrificing general modeling capabilities in other areas. The *DAT* architecture is an example of such a general architectural framework.

*In general*, both types of attention are crucial. However, this is a narrow synthetic relational task in which pairwise same/different relations are a sufficient statistic. The goal of this task is to demonstrate that the relational computational mechanisms of *DAT* translate to improvements on synthetic relational tasks compared to a Transformer, similar what is observed in previous relational architectures (see also the global response for more on this).

> In the second experiment, is a symmetrical relationship enforced? If so, is it crucial?

No, symmetric relations are not enforced in the math experiments.

> Is there any way or have the authors tried looking into attention patterns of the relational attention heads? Looking at the activations of self-attention heads has been proven to be insightful, have similar analyses been carried out for relational attention heads?

Yes. With the new large-scale language modeling experiments, we have created visualizations of the learned relational attention heads to understand both the attention pattern and the relations being modeled. We were not able to fit these figures into the 1-page pdf, but we will add some of these visualizations to the paper and will also work to make these interactive visualizations publicly available together with pretrained model checkpoints for people to explore.

We have found that the attention patterns in the relational heads in the *DAT* language models resemble the attention patterns in standard sensory heads (i.e., there are heads that seem to attend based on position, others based on syntax). The relations appear to capture both semantic and syntactic relations. For example, in the opening sentence of the Wikipedia page on finite state machines, we find that the token for "computation" has large relational activations with the token for "mathematical" and "model".

> Line 138 What are feature templates? Vectors of 0 and 1s for features to be picked and the position of 0,1 is going to be learned?

The parameter $F_{lib} = (f_1, \ldots, f_n) \in \mathbb{R}^{n_s \times d_k}$ are what we refer to as "feature templates". They are standard neural network parameters (i.e., real numbers, not 0s/1s), and are learned as part of training. Each symbol $s_i$ has a corresponding "feature template" $f_i$ which acts as its "key" in the symbolic attention operation. We call them "feature templates" because the dot product operation compares $f_i$'s with $x_j$'s, hence each $f_i$ forms a template of features that $x_i$ is matched against.
