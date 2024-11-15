
Thank you for your review. We refer you first to the global response for an overview the additions and revisions made to address all of the reviewers comments. Here, we provide a summary and then more detailed responses to your specific concerns.

**Summary of review's concerns and criticisms**
1. The experiments section does not explicitly mention parameter counts. The reviewer is concerned about the fairness of the comparisons accounting for model size since a relational attention head contains more parameters than standard self-attention.
2. The proposed Dual Attention Transformer architecture is compared only to a Transformer in the experiments.

**Summary of responses**
1. We have added explicit annotations of model parameter counts to all figures and tables. We explain below that the difference in parameter count in the current experiments is marginal. Moreover, we carried out additional experiments in which we compare against a *larger* Transformer (by increasing its model dimension). We find that *DAT* continues to outperform this baseline, confirming that the difference in performance can be attributed to the different computational mechanisms and inductive biases, rather than parameter count. (Please see the uploaded 1-page pdf for these updated results.)
2. We carried out additional experiments to compare the proposed *DAT* architecture against previously-proposed relational architectures (PrediNet, CoRelNet, Abstractor) on the relational games tasks. We find that *DAT* performs competitively with these narrow-domain architectures despite their strong task-tailored inductive biases.
3. We added new language modeling experiments with models up to 1.3B parameters, trained for 10B tokens of the Fineweb dataset. We find the *DAT* model outperforms the Transformer baseline at multiple model scales, suggesting that the model has learned to use the relational computational mechanisms in language processing. We carried out a visualization analysis of the learned relational attention heads, which will be added to the paper. This forms a challenging large-scale evaluation of the applicability of relational computational mechanisms to important complex real-world tasks.

---

**Controlling for model capacity/parameter count**

Your point about controlling for model capacity is well-taken, and we entirely agree that this is important. Please see the uploaded 1-page pdf for updated experimental results which carefully account for the effect of model capacity. For example, note that in the relational games experiments the *DAT* model is 421K parameters and is compared to Transformers with 386K parameters (same $d_{\mathrm{model}}$) and 481K parameters (increased $d_{\mathrm{model}}$). We also compare against a Transformer with a larger number of heads. Similarly, for the language modeling experiments, note that we compare a 1.31B-parameter Transformer to a 1.27B-parameter *DAT*. A similar model size comparison is made in the remaining experiments.

Note also that even without controlling for parameter count explicitly (e.g., by increasing $d_{\mathrm{model}}$), the difference in parameter count is relatively small (e.g., 421K vs 386K). We would also like to explain that the reason for emphasizing the composition of relational heads vs self-attention heads in our presentation is to assess the effect of the relational computational mechanisms. In particular, one question is whether the model can still achieve improvements in sample efficiency when the inductive bias is not strict (i.e., model contains both types of attention heads). I.e., can the model implicitly select among the computational mechanisms available to it based on the task and context? Our experiments indicate that this is indeed possible.

**Comparison to other baselines**

Please see the global response for an explanation of how our suite of experiments is designed to evaluate the empirical claims of the paper.

*Additional relational games baselines.* We have added several relational architecture baselines (PrediNet, CoRelNet, Abstractor) to the relational games experiments. We refer to Figure 1 in the 1-page pdf in the global response. We find that *DAT* performs competitively with the relational architectures, all of which outperform the Transformer.

> Basically, the experiments section is a mere ablation study comparing only variants of the proposed DAT, which includes the conventional transformer.

It is true that a standard Transformer is technically a special configuration of the *DAT* framework when the number of relational heads is zero (and this is mentioned in the paper). However, we respectfully disagree with the conclusion that this makes our experiments "a mere ablation study". The *DAT* framework is of course a novel architecture different from Transformers, with additional computational mechanisms for relational processing. As explained in the global response, the main goal of our paper is to propose a new class of architectures that supports relational computational mechanisms while preserving the generality and representational capacity of the Transformer framework (with respect to task paradigms and modalities). Thus, the experiments required to validate such claims are precisely the types of experiments we carried out in this paper and a standard Transformer is the appropriate baseline we need to compare to. 

> Why are non-Transformer baselines like, e.g. RWKV, Mamba, xLSTM missing in language modelling?

Comparison to these baselines (on, say, language modeling), while indeed interesting, would be tangential to the aim of this paper because they lack relational computational mechanisms. Our paper is specifically about extending the Transformer framework. In the context of the relational games experiments, we agree that it would be interesting to see how our framework compares with previously-proposed relational architectures with stronger inductive biases---we have added this to our experiments.

