Score: 5

Thank you for your review. We refer you first to the global response for an overview the additions and revisions made to address your concerns. Here, we first summarize our responses to your specific concerns then provide more detail.

**Summary of review's concerns and criticisms:**
- The review finds that certain terms like "relational information" can be better explained. Additionally, the motivation behind certain computational mechanisms like symbolic attention can be better explained.
- In terms of experimental evaluation, the review asks whether the conclusions drawn from the experimental results continue to hold for different model scales and architectural hyperparameters (e.g., increasing number of heads, increased model size, etc.).
- The review mentions language modeling in particular as an area where evaluation at larger scales with real-world datasets would important.

**Summary of responses.**
- We appreciate the feedback on the explanation of terminology and motivation of our proposed computational mechanisms. We will expand on the discussion in the paper with this in mind.
- We carried out additional experimental runs that evaluate different architectural hyperparameters and evaluated a range of model scales. We find that our conclusions are robust and hold across all these configurations.
- We carried out large-scale language modeling experiments with models up to 1.3B parameters in size (roughly GPT2 scale) trained on 10B tokens of high-quality text data from the Fineweb dataset. We observe improvements due to the relational computational mechanisms of *DAT* over standard Transformers at multiple model scales.

---
**On "relational information" and computational mechanisms like "symbolic attention"**

> Many parts of the work could be better motivated and better explained. Importantly, relational information is presented in a handwavy manner throughout the manuscript, without concrete examples so a reader may understand what could be/is actually captured by that term.

This point is well-taken and we appreciate this feedback. We will provide an expanded discussion in the revised manuscript, and include a preview of that here.

First, we note that formalizing the notion of "relational information" is a subtle and challenging issue in the context of distributed representations. There can be differing reasonable definitions, and different designs for computational mechanisms capturing relational information.

For us, "relational information" refers to a representation of a *comparison of the features of different objects*. The input to the model can be thought of as a sequence of object representations, and a relation between a pair of objects is a comparison of the two objects across different attributes. For example, a relation might encode information such as "same shape", "same color", "different texture", "larger than", etc. Relational attention models such comparisons via inner products. First, linear projections extract particular attributes from each object, then the inner product compares the extracted attributes.

In the relational games experiments, the relations might be across "shape" and "color". In the language modeling experiments, the relations can describe syntax (e.g., Noun-verb, subject-object, etc.) or semantics (e.g., "computation" is related to "math"), and we see some evidence that these types of relations are learned (see the response to your question on this below).

We will similarly add an expanded discussion of the motivation behind the design of symbolic attention.

---

**Expanded experiments**

> Architectural innovations, especially adaptations to Transformer should be clearly accompanied by scaling analysis along multiple axes. I.e., in relational games, what happens if more self-attention heads are provided (as opposed to the rather restrictive and extreme case of only two attention heads)? Would relational attention still be advantageous? How about a bigger model? Are these gains only marginal in specific tasks with small models? How about data-rich tasks? For these reasons, even though the experimental setup is quite diverse and comprehensive, it still does not seem so conclusive for an architectural innovation on top of the transformer architecture.

We are glad that you appreciate the diversity and comprehensiveness of our suite of experiments. Please see the global response and the uploaded 1-page pdf for additional experimental evaluations that address each of these points. We summarize below.

- Relational games ([[Fig 1]] in 1-page pdf)
    - We evaluate a Transformer baseline with increased model dimension to control for parameter count and give the Transformer an advantage in size. We also evaluate a baseline with an increased number of attention heads. The *DAT* model continues to outperform all variants of the Transformer.
    - We evaluate against previously-proposed relational architectures including PrediNet, CoRelNet, and the Abstractor. *DAT* performs competitively against these architectures despite their strong inductive biases which were designed for synthetic tasks like this.
- Mathematical problem-solving ([[Table 1]] in 1-page pdf)
    - Our initial experiments our carried out with 2-layer models. We additionally evaluate deeper and larger models, and observe that the advantage of *DAT* persists across different depths and model scales. We also evaluate a Transformer baseline with increased model dimension to control for parameter count and give the Transformer an advantage in size.
- Language modeling
    - See below.

---
**Language modeling, at scale.**

> It would have been nice if the language modelling experiments were carried out with more datasets so they would be more conclusive.

We agree. As mentioned above and in the global response, we carried out new language modeling experiments at a significantly larger scale (up to 1.3B parameters) on the Fineweb dataset. [[Table 2]] in the uploaded 1-page pdf reports the results from these experiments, showing that the relational computational mechanisms of *DAT* result in improvements in language modeling. This is an important finding and, we believe, a significant milestone in the line of work on relational architectures. So far, the success of relational architectures has been mostly limited to synthetic relational tasks (e.g., relational games). This result shows that these types of relational computational mechanisms can be integrated into a general and versatile modeling framework, granting enhanced relational processing without sacrificing general sequence modeling capabilities in other areas, and resulting in improvements even in complex real-world tasks like language understanding.

---

**Questions.**
> Why relational-only attention in Figure 2 works best? Any intuitions for that? Aren’t the two types of attention crucial (according to the intro)? Why can we get away with only relational attention?

This is an important question. Yes, there is a clear intuitive reason for this. The relational games tasks are "purely relational" tasks in the sense that each task is entirely solvable with only the pairwise same/different relations. Relational attention can capture this information more efficiently than standard (sensory) attention. This is also why the standard Transformer is less efficient on this task. In fact, based on prior, what is interesting about the result is actually that the configuration with both types of attention heads is *not* significantly worse. A common idea in previous work is that relational learning requires strict inductive biases that "filter out" non-relational information because it would corrupt the relational information and require large amounts of data to overcome. This result demonstrates that merely offering the model different computational mechanisms, including relational computational mechanisms, is sufficient and that the model could learn use the appropriate mechanism based on the task and context. This is important because it suggests the possibility of imbuing a general modeling framework with enhanced relational processing capabilities without sacrificing general modeling capabilities in other areas. The *DAT* architecture is an example of such a general architectural framework.

*In general*, both types of attention are crucial. However, this is a narrow synthetic relational task in which pairwise same/different relations are a sufficient statistic. The goal of this task is to demonstrate that the relational computational mechanisms of *DAT* translate to improvements on synthetic relational tasks compared to a Transformer, similar what is observed in previous relational architectures (see also the global response for more on this).

> In the second experiment, is a symmetrical relationship enforced? If so, is it crucial?

No, symmetric relations are not enforced in the math experiments.

> Is there any way or have the authors tried looking into attention patterns of the relational attention heads? Looking at the activations of self-attention heads has been proven to be insightful, have similar analyses been carried out for relational attention heads?

Yes. With the new large-scale language modeling experiments, we have created visualizations of the learned relational attention heads to understand both the attention pattern and the relations being modeled. We were not able to fit these figures into the 1-page pdf, but we will add some of these visualizations to the paper and will also work to make these interactive visualizations publicly available together with pretrained model checkpoints for people to explore.

We found that the attention patterns in the relational heads in the *DAT* language models resemble the attention patterns in standard sensory heads (i.e., there are heads that seem to attend based on position, others based on syntax). The relations appear to capture both semantic and syntactic relations. For example, in the opening sentence of the wikipedia page on finite state machines, we find that the token for "computation" has large relational activations with the token for "mathematical" and "model".

> Line 138 What are feature templates? Vectors of 0 and 1s for features to be picked and the position of 0,1 is going to be learned?

The parameter $F_{lib} = (f_1, \ldots, f_n) \in \mathbb{R}^{n_s \times d_k}$ are what we refer to as "feature templates". They are standard neural network parameters (i.e., real numbers, not 0s/1s). Each symbol $s_i$ has a corresponding "feature template" $f_i$ which acts as its "key" in the symbolic attention operation. We call them "feature templates" because the dot product operation compares $f_i$'s with $x_j$'s, hence each $f_i$ forms a template of features that $x_i$ is matched against.

---
---

[Need to cut down to 6K characters, or move some responses to "official comment" rather than "rebuttal". E.g., questions can be in official comment.]