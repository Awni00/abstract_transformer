Thank you for your review. We refer you first to the global response for an overview the additions and revisions made to all reviewer comments. Here, we first summarize our responses to your specific concerns then provide more detail.

**Summary of review's concerns and criticisms**
- The review suggests that certain terms like "relational information" can be better explained. Additionally, the motivation behind certain computational mechanisms like symbolic attention can be better explained.
- In terms of experimental evaluation, the review asks whether the conclusions drawn from the experimental results continue to hold for different model scales and architectural hyperparameters (e.g., increasing number of heads, increased model size, etc.).
- The review mentions language modeling in particular as an area where evaluation at larger scales with real-world datasets would be important.

**Summary of responses**
- We appreciate the feedback on the explanation of terminology and motivation of our proposed computational mechanisms. We will expand on the discussion in the paper with this in mind.
- We carried out additional experimental runs that evaluate different architectural hyperparameters and evaluated a range of model scales. We find that our conclusions are robust and hold across all these configurations.
- We carried out large-scale language modeling experiments with models up to 1.3B parameters in size (roughly GPT2 scale) trained on 10B tokens of high-quality text data from the Fineweb dataset. We observe improvements due to the relational computational mechanisms of *DAT* over standard Transformers at multiple model scales.

**On "relational information" and computational mechanisms like "symbolic attention"**

> Many parts of the work could be better motivated and better explained. Importantly, relational information is presented in a handwavy manner throughout the manuscript, without concrete examples so a reader may understand what could be/is actually captured by that term.

This point is well-taken and we appreciate this feedback. We will provide an expanded discussion in the revised manuscript, and include a preview of that here.

First, we note that formalizing the notion of "relational information" is a subtle and challenging issue in the context of distributed representations. There can be differing reasonable definitions, and different designs for computational mechanisms capturing relational information.

For us, "relational information" refers to a representation of a *comparison of the features of different objects*. The input to the model can be thought of as a sequence of object representations, and a relation between a pair of objects is a comparison of the two objects across different attributes. For example, a relation might encode information such as "same shape", "same color", "different texture", "larger than", etc. Relational attention models such comparisons via inner products. First, linear projections extract particular attributes from each object, then the inner product compares the extracted attributes.

In the relational games experiments, the relations might be across "shape" and "color". In the language modeling experiments, the relations can describe syntax (e.g., Noun-verb, subject-object, etc.) or semantics (e.g., "computation" is related to "math"), and we see some evidence that these types of relations are indeed learned (see the response to your question on this below).

We will similarly add an expanded discussion of the motivation behind the design of symbolic attention.

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

**Language modeling at greater scale**

> It would have been nice if the language modelling experiments were carried out with more datasets so they would be more conclusive.

We agree. As mentioned above and in the global response, we carried out new language modeling experiments at a significantly larger scale (up to 1.3B parameters) on the Fineweb dataset. [[Table 2]] in the uploaded 1-page pdf reports the results from these experiments, showing that the relational computational mechanisms of *DAT* result in improvements in language modeling. This is an important finding and, we believe, a significant milestone in the line of work on relational architectures. So far, the success of relational architectures has been mostly limited to synthetic relational tasks (e.g., relational games). This result shows that these types of relational computational mechanisms can be integrated into a general and versatile modeling framework, granting enhanced relational processing without sacrificing general sequence modeling capabilities in other areas, and resulting in improvements even in complex real-world tasks like language understanding.



