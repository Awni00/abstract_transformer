Score: 4

**Summary of Criticism:**
There is less concrete things to respond to in this review. But...
- There is a recurring claim that DAT is worse than the Abstractor on the math experiments. Need to understand where that is coming from and respond to it.
- Asked why PrediNet not in baselines for relational games. Can add this as a baseline. (Together with some others)

> The experiments in this work are insufficient. For example, the experimental results in fig.3 are overall inferior to the Abstractor. Since the computational mechanisms of DAT and Abstractor are similar, the authors should provide more reasons or explanations for such performance gaps.

- The Abstractor was not among the baselines reported in our experiments, as we only compared to a Transformer. It seems that this assertion is perhaps based on examining the figures in our paper and the Abstractor paper by eye. We highlight that is a [flawed] way to make a comparison as there may exist several factors that differ among the respective experimental set ups, including model size, data splits, optimizer hyperparameters, etc.
- We also highlight that our empirical evaluation included a suite of experiments across different data modalities and task paradigms: relational tasks (relational games), mathematical problem-solving, image recognition (ImageNet), and language modeling (Tiny Stories, and now on a larger scale on Fineweb with upto 1.3B-parameter models). We note that the Abstractor was not evaluated on this range of experiments in the original paper and does not support task paradigms such as language modeling. Hence, a comparison to the Abstractor on one set of experiments (moreover, in an uncontrolled environment) does not provide a meaningful assessment of our work on the basis of its stated aims.'
- [TODO] Nonetheless, we carried out additional experiments to evaluate the Abstractor on the mathematics experiments in the same controlled environment to enable making a comparison. [Describe ...]

TODOs:
>In fig.6 match pattern and fig.12 match pattern, Dual-Attn Transformer (nsa = 1, nra = 1) performs differently, why?
- Fig 6 explores the effect of a symmetry inductive bias in the relations inside relational attention on the relational games benchmark. That is, it compares the learning curves of matching dual-attention Transformer models, one with a symmetry constraint on the relations in RA ($W_q^{rel} = W_k^{rel}$) and the other with no symmetry constraint. Figure 12 evaluates the performance of a Dual Attention Transformer variant with the Abstractor's RCA instead of relational attention as proposed in this work. In Figure 12, neither model has a symmetry constraint. Thus, the solid lines in Figure 12 match the dashed lines in Figure 6, since the models in Fig 12 have no symmetry constraint. We will add another ablation figure which compares relational attention against the Abstractor's RCA *with* a symmetry constraint on the relations (in the case of the Abstracor where the relations and the attention scores are the same thing, we will take 'symmetry' to mean symmetry over the attention scores).


"We kindly ask that you clarify what you mean by this, or let us know whether our responses address this concern, as we are unsure what you mean by this."