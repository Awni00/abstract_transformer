Score: 4

Thank you for your review. We refer you first to the global response for an overview the additions and revisions made to address your concerns. Here, we first summarize our responses to your specific concerns then provide more detail.

**Summary of review's concerns and criticisms:**
- This review is concerned with how the proposed architecture compares with the Abstractor, a related architecture proposed in previous work. The review focuses on the mathematics experiments which also appeared in the Abstractor paper and asserts that our model does not perform as well.
- The reviewer also asks about comparisons to PrediNet as a baseline for the relational games experiments.

**Summary of responses.**
- The review appears to be comparing the *DAT* model reported in our experiments to a much larger model in the Abstractor paper. We carried out additional experiments at larger scales and found that *DAT* compares favorably to the Abstractor when controlling for scale.
- We added several relational architecture baselines to the relational games experiments, including PrediNet, CoRelNet, and Abstractor models.

---

**Comparison between *DAT* and the Abstractor (on math).**
> The experiments in this work are insufficient. For example, the experimental results in fig.3 are overall inferior to the Abstractor.

<!-- Since the computational mechanisms of DAT and Abstractor are similar, the authors should provide more reasons or explanations for such performance gaps. -->

> In fig.3, except for algebra__sequence_next_term, the four types of mathematical problem-solving tasks perform poorly compared to Abstractors. It seems that the dual attention mechanism did not work well.

We are unsure why you think that the experimental results in Figure 3 are inferior to the Abstractor. The Abstractor paper also evaluates on the mathematics dataset considered in this work. Perhaps you are comparing Fig. 3 in our paper to Fig. 7 in the Abstractor paper? Please let us know.

If so, the results of the two papers are not comparable since they are under different experimental settings. In particular, *the models in the Abstractor paper are much larger than the ones reported in Figure 3 in the current version of our paper*. The models in the Abstractor paper are roughly 2 million parameters in size (2.3M for Transformer and 2.2M for Abstractor). By contrast, in the current version of the paper, our models are roughly 800K parameters. (Note that the large model size of the Abstractor despite modest-seeming hyperparameters is due to the "multi-attention decoder" in the architecture that is used to attend alternatingly to the Encoder and Abstractor.)

We carried out additional experiments scaling up our models on the math experiments. In particular, while our current models are 2-layers deep, we carried out additional experimental runs with 3 layers (1M-parameter scale) and 4 layers (1.5M-parameter scale). These new results are depicted in [[Table 1]] of the uploaded 1-page pdf. When comparing, the 1.5M-parameter scale *DAT* models to the results reported in the Abstractor paper (which is 2.2M parameter), *DAT* compares favorably.

In addition to allowing for comparison to the Abstractor's results at comparable scales, these new experiments also demonstrate that the superiority of *DAT* over a standard Transformer consistently holds across models of varying sizes and depths.

Thus, these new results demonstrate that, when compared at similar model sizes, *DAT* perform favorably compared to both the Abstractor and a standard Transformer. Moreover, we emphasize that the math experiments are merely one set of experiments among a suite of experiments consisting of four distinct set of experiments across different modalities and task paradigms. In particular, our *DAT* architecture is more general than the Abstractor (and, we argue, more natural, versatile, and scalable) and supports decoder-only language modeling---an important task paradigm that the Abstractor does not support. More on this below.

As we will discuss below, we have also added the Abstractor as a baseline in the relational games experiments.

---

**Additional baselines like PrediNet.**
> In fig.2, why wasn’t PrediNet introduced for comparison? Since PrediNet is the baseline of relational games and has an attention mechanism.

We have carried additional experimental runs for the relational games experiments evaluating learning curves on baselines including PrediNet, CoRelNet, and the Abstractor. We also add additional Transformer baselines increasing model size and number of attention heads to further test the claim that *DAT* outperforms Transformers on relational tasks.

We find that all relational architectures generally perform favorably compared to the Transformer baselines in terms of sample-efficiency. In our experiments, *DAT* outperforms PrediNet and the Abstractor in 4 out 5 relational games tasks. Expectedly, CoRelNet performs strongly on this task. (CoRelNet has very strong inductive biases that encode the same/different relations that are a sufficient statistic for the relational games tasks, and was designed with benchmarks like this in mind.) Nonetheless, we observe that *DAT* compares favorably to relational architectures on synthetic relational tasks despite being a far more general and versatile modeling framework.

---

**The goal of our paper and how our suite of experiments evaluate our empirical claims.**

> The experiments in this work are insufficient.

We point first to the global response, where we explain the main goals and claims of the paper, and how the suite of experiments is designed to evaluate these claims.

In particular, we would like to highlight the new large-scale language modeling experiments that we added. Previously, our language modeling experiments consisted of modest-size models trained on the synthetically (LLM-)generated Tiny Stories dataset. We have greatly expanded our evaluation of *DAT* as a language model. In particular, we now evaluate models up to 1.3B-parameters in scale trained on the fineweb dataset (a real-world dataset of high-quality text scraped from the internet). [[Table 2]] in the uploaded 1-page pdf reports the results from these experiments, showing that the relational computational mechanisms of *DAT* result in improvements in language modeling. This is an important finding and, we believe, a significant milestone in the line of work on relational architectures. So far, the success of relational architectures has been mostly limited to synthetic relational tasks (e.g., relational games). This result shows that these types of relational computational mechanisms can be integrated into a general and versatile modeling framework, granting enhanced relational processing without sacrificing general sequence modeling capabilities in other areas, and resulting in improvements even in complex real-world tasks like language understanding.

In addition to the language modeling experiments, our suite of experiments span multiple task paradigms and data modalities, evaluating the proposed architecture in a wide range of domains.

---
**Misc questions/comments.**

> In Eq.3, does SymbolRetriever mean Symbol Assignment Mechanisms?
Yes. The SymbolRetriever returns a sequence of "symbols" $(s_1, ..., s_n)$, via one of the symbol assignment mechanisms: positional symbols, position-relative symbols, or symbolic attention (though in the case of the latter the computation is of course implemented differently than the first two). We will make this clear in the revised text.

>In fig.6 match pattern and fig.12 match pattern, Dual-Attn Transformer (nsa = 1, nra = 1) performs differently, why?

Fig 6 explores the effect of a symmetry inductive bias in the relations inside relational attention on the relational games benchmark. That is, it compares the learning curves of matching dual-attention Transformer models, one with a symmetry constraint on the relations in RA ($W_q^{rel} = W_k^{rel}$) and the other with no symmetry constraint. Figure 12 evaluates the performance of a Dual Attention Transformer variant with the Abstractor's RCA instead of relational attention (proposed in this work). In Figure 12, neither model has a symmetry constraint. Thus, the solid lines in Figure 12 match the dashed lines in Figure 6, since the models in Fig 12 have no symmetry constraint. We will add another ablation figure which compares relational attention against the Abstractor's RCA *with* a symmetry constraint on the relations (in the case of the Abstracor where the relations and the attention scores are the same thing, we will take 'symmetry' to mean symmetry over the attention scores).


---
---
Below this line are old rough notes.

Currently, roughly 8.6K characters. Need to get down to 6K characters. One way to achieve this is by relegating the **Misc questions/comments** section to an "official comment" rather than a "rebuttal". Reviewers are not "required" to read those, but perhaps that's fine (we don't really know if they're gonna read the rebuttal either. The questions are not the main reason for their score, the other stuff is more important to cover.)

**Summary of response:**
- The Abstractor was not reported as a baseline in the first version of our paper. The comparison the reviewer draws appears to be by comparing the results reported in the Abstractor paper and this one. We note that this comparison is not valid since the model sizes in our version of the experiment are significantly smaller.
- We added an Abstractor baseline to the math experiments to enable a direct comparison between the two architectures, in the same controlled environment. We find that [CONTINUE...]
- We highlight appendix [XXX], in which we provide a detailed discussion comparing the computational mechanisms in the two architectures as well as carrying several comparisons comparing the relational attention mechanism proposed in this paper and the RCA mechanism of the Abstractor. We also highlight, as discussed in that section, that the Abstractor is a less general architecture that can only be applied to a narrower set of task paradigms (e.g., cannot be directly applied to language modeling).
<!-- - Finally, we explain how our suite of experiments support the technical claims of the paper -->

There is less concrete things to respond to in this review. But...
- There is a recurring claim that DAT is worse than the Abstractor on the math experiments. Need to understand where that is coming from and respond to it.
- Asked why PrediNet not in baselines for relational games. Can add this as a baseline. (Together with some others)

> The experiments in this work are insufficient. For example, the experimental results in fig.3 are overall inferior to the Abstractor. Since the computational mechanisms of DAT and Abstractor are similar, the authors should provide more reasons or explanations for such performance gaps.

- The Abstractor was not among the baselines reported in our experiments, as we only compared to a Transformer. It seems that this assertion is perhaps based on examining the figures in our paper and the Abstractor paper by eye. We highlight that is a [flawed] way to make a comparison as there may exist several factors that differ among the respective experimental set ups, including model size, data splits, optimizer hyperparameters, etc.
- We also highlight that our empirical evaluation included a suite of experiments across different data modalities and task paradigms: relational tasks (relational games), mathematical problem-solving, image recognition (ImageNet), and language modeling (Tiny Stories, and now on a larger scale on Fineweb with upto 1.3B-parameter models). We note that the Abstractor was not evaluated on this range of experiments in the original paper and does not support task paradigms such as language modeling. Hence, a comparison to the Abstractor on one set of experiments (moreover, in an uncontrolled environment) does not provide a meaningful assessment of our work on the basis of its stated aims.'
- [TODO] Nonetheless, we carried out additional experiments to evaluate the Abstractor on the mathematics experiments in the same controlled environment to enable making a comparison. [Describe ...]

TODOs:


"We kindly ask that you clarify what you mean by this, or let us know whether our responses address this concern, as we are unsure what you mean by this."