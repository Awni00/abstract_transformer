Thank you for your review. We refer you first to the global response for an overview of the additions and revisions made to address all of the reviewers comments. Here, we give detailed responses to your specific concerns.


**On "relational information" and computational mechanisms like "symbolic attention"**

> Many parts of the work could be better motivated and better explained. Importantly, relational information is presented in a handwavy manner throughout the manuscript, without concrete examples so a reader may understand what could be/is actually captured by that term.

This point is well-taken and we appreciate this feedback. We will provide an expanded discussion in the revised manuscript, and include a preview of that here.

First, we note that formalizing the notion of "relational information" is a subtle and challenging issue in the context of distributed representations. There can be differing reasonable definitions, and different designs for computational mechanisms capturing relational information.

For us, "relational information" refers to a representation of a *comparison of the features of different objects*. The input to the model can be thought of as a sequence of object representations, and a relation between a pair of objects is a comparison of the two objects across different attributes. For example, a relation might encode information such as "same shape", "same color", "different texture", "larger than", etc. The relational attention mechanism models such comparisons via inner products. First, linear projections extract particular attributes from each object, then the inner product compares the extracted attributes.

In the relational games experiments, the task-relevant relations might encode comparisons of "shape" and "color". In the language modeling experiments, the relations can describe syntax (e.g., Noun-verb, subject-object, etc.) or semantics (e.g., "computation" is related to "math"), and we see some evidence that these types of relations are indeed learned (see the response to your question on this below).

We will similarly add an expanded discussion of the motivation behind the design of symbolic attention.

**Expanded experiments**

> Architectural innovations, especially adaptations to Transformer should be clearly accompanied by scaling analysis along multiple axes. I.e., in relational games, what happens if more self-attention heads are provided (as opposed to the rather restrictive and extreme case of only two attention heads)? Would relational attention still be advantageous? How about a bigger model? Are these gains only marginal in specific tasks with small models? How about data-rich tasks? For these reasons, even though the experimental setup is quite diverse and comprehensive, it still does not seem so conclusive for an architectural innovation on top of the transformer architecture.

We are glad that you appreciate the diversity and comprehensiveness of our suite of experiments. Please see the global response and the uploaded 1-page pdf in the global response for additional experimental evaluations that address each of these points. We summarize below.

- Relational games (Figure 1 in 1-page pdf)
    - We evaluate a Transformer baseline with increased model dimension to control for parameter count and give the Transformer an advantage in size. We also evaluate a baseline with an increased number of attention heads. The *DAT* model continues to outperform all variants of the Transformer.
    - We evaluate against previously-proposed relational architectures including PrediNet, CoRelNet, and the Abstractor. *DAT* performs competitively against these architectures despite their strong inductive biases which were designed for synthetic tasks like this.
- Mathematical problem-solving (Table 1 in 1-page pdf)
    - Our initial experiments our carried out with 2-layer models. We additionally evaluate deeper and larger models, and observe that the advantage of *DAT* persists across different depths and model scales. We also evaluate a Transformer baseline with increased model dimension to control for parameter count and give the Transformer an advantage in size.
- Language modeling
    - See below.

**Language modeling at greater scale**

> It would have been nice if the language modelling experiments were carried out with more datasets so they would be more conclusive.

We agree. As mentioned above and in the global response, we carried out new language modeling experiments at a significantly larger scale (up to 1.3B parameters) on the Fineweb dataset. Table 2 in the uploaded 1-page pdf reports the results from these experiments, showing that the relational computational mechanisms of *DAT* result in improvements in language modeling. This is an important finding and, we believe, a significant milestone in the line of work on relational architectures. So far, the success of relational architectures has been mostly limited to synthetic relational tasks (e.g., relational games). This result shows that these types of relational computational mechanisms can be integrated into a general and versatile modeling framework, granting enhanced relational processing without sacrificing general sequence modeling capabilities in other areas, and resulting in improvements even in complex real-world tasks like language understanding. Moreover, our analysis of the relations learned by the relational heads in the *DAT* model suggests that they encode meaningful and interpretable semantic relations.

---

# Discussion: Response 1

> I would like to thank the authors for their efforts, for carrying out more experiments, and for their clarifications. In particular, I find the examples and clarifications for relational information quite helpful and compelling, and would encourage the authors to include them in the revised manuscript.
>>
>A number of my concerns were addressed including increasing the scale of experiments and using more self-attention heads, and I thank the authors for that.
>
>A few concerns still remain and I'd appreciate it if the authors could comment on those.
>
>I am still curious about the answer to my last question, i.e., I can't figure out why self-attention still should not overwhelm the relational information when the information is merely concatenated, are there any other mechanisms (like small $d_r$) in play to help with that? Any intuitions on why the sensory information doesn't dominate the flow of information?
>Concern w.r.t. sensitivity (asked in the original review): I understand that some of the experiments like the language modeling are large-scale, and might be difficult to be carried out multiple times, but other than the synthetic experiments, I can't see any measure of the reliability of the results. Do they hold if we train with different initializations? Also, have you extensively searched for HPs for DAT? What about for transformer? These details can make a difference in one's take on the proposed method. Overall, while I like the generality of the method and the diversity of the benchmarks, I'm still concerned about the reliability of the results (especially the differences in language modeling tasks are not often huge for one to feel compelled with just one run).
>Although not probably feasible for now, I'd definitely encourage the authors to carry out similar extensions on experiments with VITs, i.e., more heads, averaging over a number of seeds, etc. to add to the validity of their proposal.
>As a side note, the visualization of attention maps is sadly not correctly appearing in the pdf. Also, it'd be nice if patterns similar to the ones described by the authors in their rebuttal as the motivation could be found to further make a concrete case for the usefulness of the inductive bias in question.
>
>Once again I thank the authors for their efforts and clarifications. I will maintain my score for now, and will adjust it based on further discussions with the authors and other reviewers.

---


Thank you very much for your response and for engaging with our work. We are glad that you found our additional experiment and our explanations compelling, and are delighted that a number of your concerns were addressed. We are eager to address each of your remaining concerns.

**Visualization of relational attention.**
> As a side note, the visualization of attention maps is sadly not correctly appearing in the pdf. Also, it'd be nice if patterns similar to the ones described by the authors in their rebuttal as the motivation could be found to further make a concrete case for the usefulness of the inductive bias in question.

We are unsure why the visualizations are not appearing on the pdf for you. Could you please try a different pdf viewer and let us know if that fixes things for you? We are able to view the images through the chrome pdf viewer and adobe acrobat. If you are still unable to view it, perhaps we can share it an alternative manner (e.g., upload the figure somewhere and post a link; though we will need to do that through the AC to confirm anonymity is maintained as per the Neurips instructions).

In the meantime, perhaps we can briefly describe the content of the visualization for you. The figure visualizes the relations in relational attention (i.e., $r_{ij}$ in Alg 1) in a *DAT* language model trained on 10B tokens of the Fineweb dataset. The language model is applied to the following sentence: "A finite-state machine (FSM) or finite-state automaton (FSA, plural: automata), finite automaton, or simply a state machine, is a mathematical model of computation." (This is the opening sentence of the wikipedia page on FSMs, which we chose arbitrarily.) The figure depicts the relation activations in layer 0 (i.e., the first attention layer following the embedding layer) and layer 11 (half-way through the 24-layer model; keep in mind that each token at this layer contains contextual information as well). In layer 0, we focus on the `[computation]` token, and see strong relation activations to the `[mathematical]` and `[model]` tokens, as well as the `[ata]` and `[aton]` tokens in "automata" and "automaton", respectively. In layer 11, we focus on the `[mathematical]` token, and observe strong activations with the tokens for `[machine]`, `[state]`, and `[computation]`. The remaining tokens have weak relation activations. This suggests that the model learns to represent *semantic relations* inside of the relational attention mechanism.

**The balance between sensory and relational information**
>I am still curious about the answer to my last question, i.e., I can't figure out why self-attention still should not overwhelm the relational information when the information is merely concatenated, are there any other mechanisms (like small $d_r$) in play to help with that? Any intuitions on why the sensory information doesn't dominate the flow of information?

Thank you for raising this question again, and apologies for missing it in our initial response. It's an interesting question. There are several factors that may be relevant to your question.

- There exists learned maps that control which part of the embedding vector (i.e., residual stream) to "read from" or "write to", for both relational and sensory information, at each layer. For example, the $W_o^{sa}$ and $W_o^{ra}$ maps control where to write to and $W_q, W_k, W_v$ maps control where to read from. Our intuition on this is shaped by recent "mechanistic interpretability" work (for example, the "Transformer Circuits" thread, by Anthropic). In other words, the architecture contains mechanisms for reading from and writing from different subspaces of the residual stream, and is able to separate or combine sensory and relational information as needed.
- Within the relational heads, the relational information $r_{ij}$ is only directly combined with "symbolic information", not sensory information. That is, $r_{ij}$ is combined with $s_j$ via the learned $W_r$ and $W_s$ maps. Here, again, there exists mechanisms for controlling how to combine relational and symbolic information.
- Note that relative size of the sensory vector $e_i$ and relational vector $a_i$ in Dual Attention (Algorithm 1) is proportional to the number of sensory and relational heads. For example, if $n_h^{sa} = n_h^{ra}$, then $e_i$ and $a_i$ are each of dimension $d_{\mathrm{model}} / 2$. Thus, there is a balance between sensory information and relational information, in terms of the number of dimensions that correspond to each in the embedding vector.

Interestingly, one intuition argued for by certain previous works on relational architectures is that there needs to be a *strict separation* between sensory and relational information, otherwise relational information would be corrupted or overwhelmed by sensory information. Perhaps this is partly where your concern about sensory information overwhelming relational information stems from. We started off with a similar intuition, but this changed over the course of this project. In our architecture, sensory and relational information can be separated in principle via an appropriate choice of read-in and read-out parameters (as discussed above), but the model also has the flexibility to combine and compose this information. We find that, as long as there exists mechanisms for separating and combining the different types of information, the model can learn to use both types of computational mechanisms available to it. In fact, an important aspect of the architecture, is the gain in representational power that this flexibility allows.

*Gain in representational capacity by allowing sensory and relational information to interact.* Related to your question, we would like to highlight an important property of the *DAT* architecture. In an architecture where sensory information and relational information are strictly separated, the two types of computations cannot be composed. By allowing for sensory computation and relational computation to be composed, we gain a new powerful type of computation. For example, we can compose relational computation after sensory computation, where self-attention first combines the sensory information of multiple tokens in the context, and relational attention then computes relations between these combined sensory representations. Similarly, we can compose sensory computation after relational computation, where self-attention retrieves sensory information computed at a different token at an earlier layer. This is naturally and directly supported by dual attention and the *DAT* architecture, and makes a powerful new type of computation available to the model.

**Sensitivity to initialization and hyperparameter choice**
> ... I understand that some of the experiments like the language modeling are large-scale, and might be difficult to be carried out multiple times, but other than the synthetic experiments, I can't see any measure of the reliability of the results. Do they hold if we train with different initializations? Also, have you extensively searched for HPs for DAT? What about for transformer? ...

We appreciate this question, and agree on its importance. We made various efforts to validate our results with respect to sensitivity to initialization and choice of hyperparameters.

*Sensitivity to random initialization.*
- The learning curves estimated in the Relational Games experiments are carried out with several trials with different random seeds, and our results are reported with bootstrap 95\% confidence intervals. You can see this in the original figure in the paper (Fig 2) and also in the updated figure in the uploaded 1-page pdf. The difference in performance is well above the margin of error.
- The mathematical problem-solving experiments are also carried out with several trials and reported with bootstrap 95\% confidence intervals. Here, too, the difference in performance is well above the margin of error.
- As you mentioned, for the large-scale language modeling and image recognition experiments, performing multiple trials becomes computationally challenging/prohibitive. We appreciate your acknowledgment of this. Nonetheless, we will work to carry out a small number of trials for these larger scale experiments (including for the ViT experiments, as you suggested). We note that in our initial experimentation, we observe that there is a remarkably small variation in training curves over different random initialization when the architecture and training procedure are fixed. (A possible theoretical explanation for this comes from the line of work on "Tensor Programs" by Greg Yang and coauthors.) This makes us confident that the difference in performance is not due to random initialization.

*Experiment tracking, reproducibility, and transparency.* We used experiment tracking (through "Weights & Biases") for all our experiments, and all experimental logs will be made publicly available in the de-anonymized version of the paper. In addition to metrics tracked throughout training, this will allow exact reproducibility by making all aspects of the experimental configuration (e.g., git commit ID, all hyperparameter details, exact script arguments, etc) transparent. This will also include additional results and metrics that were not included in the paper due to space constraints.


*Hyperparameter choice.*
The choice of hyperparameters can be divided into those that are shared between Transformers and *DAT*, and those that are specific to *DAT* due to the additional computational mechanisms.
- For shared hyperparameters (e.g., $d_{\mathrm{model}}$, \# of layers, etc.), we use common scalings of hyperparameters, without further individual hyperparameter tuning. For example, the hyperparameters for our 350M-scale and 1.3B-scale language model experiments, our hyperparameters are based based on GPT-3 Medium (350M) and GPT-3 XL (1.3B), respectively. These standard hyperparameter choices are optimized to be good scaling choices for standard Transformer architectures through previous experimental work. Similarly, our optimization hyperparameters (e.g., learning rate schedule, weight decay, etc.) are based on common training recipes and are not tuned further.
- For *DAT*-specific hyperparameters, such as the composition of self-attention vs relational heads, the type of symbol assignment mechanism, etc., we experiment with different choices to understand the effects of different computational mechanisms and hyperparameters within the *DAT* architecture. In such cases, we present these results in the paper and discuss the observed effects of different hyperparameters (e.g., see Appendix C and D).
- In our updated and expanded experiments, we carry out experiments with additional configurations of the Transformer baseline to further validate that the gains we observe are attributable to the computational mechanisms of the *DAT*. In particular, we increase and vary the model dimension and number of heads of the Transformer baseline in both the Relational Games and mathematical problem-solving experiments (in a sense, giving it an advantage over the *DAT* model). These changes have a marginal effect, and the trends from our experimental results continue to hold.

In summary, we agree about the importance of uncertainty quantification and reproducibility in machine learning research at large, and took several steps to validate our experimental results with respect to these factors.

---

Please let us know whether this addresses your concern or if you have any remaining questions or concerned. We thank you again for the discussion.


# Response 2

Thank you for your response. We greatly appreciate the discussion.

We are glad that your concern around hyperparameter selection has been resolved. Below, we would like to respond to your follow up questions on sensitivity to hyperparameters/random initialization, as well as your question on intuitions regarding the integration of sensory and relational information.

***Sensitivity to hyperparameters in large-scale experiments***

We understand and appreciate your skepticism regarding sensitivity to hyperparameters and random initialization in the large-scale experiments, like language modeling. We also appreciate your recognition of the technical difficulties associated with carrying out multiple trial runs for experiments at this scale.

Towards addressing this concern, we carried out additional trial runs with slightly different hyperparameters for the *DAT* model. Below is the updated results table for the language modeling experiments.

| Model | Param Count | $d_{\mathrm{model}}$ | $n_{\mathrm{layers}}$ | $n_h^{sa}$ | $n_h^{ra}$ | $d_r$ | $n_h^{kv}$ | $n_s$ | $n_h^{sym}$ | Perplexity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|:---:|
| Transformer | 353M | 1024 | 24 | 16 | 0 | - | - | - | - | 16.94 |
| DAT | 368M | 1024 | 24 | 8 | 8 | 32 | - | 1024 | 8 | 15.97 |
| DAT | 343M | 1024 | 24 | 8 | 8 | 8 | 4 | 1024 | 8 | 16.26 |
| DAT | 330M | 1024 | 24 | 8 | 8 | 8 | 2 | 1024 | 8 | 16.42 |
| DAT | 325M | 1024 | 24 | 8 | 8 | 8 | 1 | 1024 | 8 | 16.44 |
|
| Transformer | 1.31B | 2048 | 24 | 32 | 0 | - | - | - | - | 13.63 |
| DAT | 1.37B | 2048 | 24 | 16 | 16 | 64 | - | 2048 | 8 | 13.43 |
| DAT | 1.27B | 2048 | 24 | 16 | 16 | 64 | 8 | 512 | 16 | 13.44 |
| DAT | 1.27B | 2048 | 24 | 16 | 16 | 64 | 8 | 1024 | 8 | 13.49 |
| DAT | 1.27B | 2048 | 24 | 16 | 16 | 64 | 8 | 2048 | 8 | 13.45 |

At the 350M-parameter scale, we carry out additional experiments where we vary hyperparameters to vary the model size. In particular, we add results for 330M-parameter and 325M-parameter DAT models. We find that these DAT models continue to significantly outperform the Transformer baseline despite being up to 8% smaller. On the 1.3B-parameter scale, we experiment with the hyperparameters of symbolic attention (e.g., varying the number of symbols in the symbol library and the number of heads in symbolic attention). Again, we observe that all experimental runs consistently outperform the Transformer baseline and have relatively little variation across different configurations.

**These results demonstrate that the performance gains due to the relational computational mechanisms in DAT are robust to variations in hyperparameters and different random initialization across experimental runs.**

We will work to carry out additional trial runs for the final version of the paper, including for the image recognition experiments as well. We also plan to make this investigation more systematic.

We hope this helps to address your concern regarding sensitivity to hyperparameters.

---

***Visualization***

> if I understand correctly, the pdf also contained one example. ... I'd just add that please do not rely on one example and provide sufficient evidence in your final version.

Yes, the pdf included just one example. This is simply due to space constraints since we are only allowed a single-page pdf. Of course, we entirely agree that one example is insufficient to draw reliable conclusions about the types of relational representations learned by relational attention in DAT. The final version of the paper will include a section dedicated to these visualizations which will contain multiple examples. We also plan to create an interactive visualization tool where users can enter their own prompt text and visualize the full network (all layers, all relations, etc.).

---

***Intuition on the balance between sensory and relational information***

In your question, you expressed an intuition that sensory information might "overwhelm" relational information. In our previous response, we aimed to share our own intuitions around how sensory and relational information are integrated in the DAT architecture. It seems we did not fully understand your question, however. We hope to elaborate here.

First, we preface by saying that intuitions can be misleading in neural network architecture design. The ultimate test of an idea is empirical evaluation. Our empirical evaluation demonstrates the effectiveness of the relational computational mechanisms in the DAT architecture, and show an ability to integrate sensory and relational information.

Nonetheless, we would like to attempt to explain our intuition and provide some more discussion that might help address your question.

Addressing this question requires being more specific about the underlying architecture and the mechanism it uses for integrating sensory and relational information. This determines *how* and *why* sensory information might overwhelm relational information in some situations. (And, indeed, what precisely it would mean for sensory information to "overwhelm" relational information.)

The key concrete example for the purposes of our discussion is how sensory and relational information is integrated in self-attention in standard Transformers. In self-attention the attention scores, which can be understood as relations, guide information flow by weighting value vectors that encode object features. This interpretation is presented in Altabaa et al. 2024 as a motivation for the proposal of the Abstractor. Thus, technically, relational information is present in the representations produced by self-attention. However, because the low-dimensional attention relations are multiplied with high-dimensional sensory information, they are *entangled*, making the underlying relations very difficult to recover.

One way to think about how the Abstractor addresses this entanglement is that it "downgrades" the sensory information by replacing the object features with "symbols". Thus, although the attention relations are still being multiplied with the "values", the values are now quasi-symbolic vectors with low-variance with respect to object features, making relational information more directly recoverable. By contrast, our proposal in this work can be understood as **"upgrading" the relational information to be on equal footing with the sensory information**. DAT achieves this in relational attention by making the *values being retrieved by attention to be relations, themselves*.

In our architecture, **sensory and relational information are on equal footing.** Concatenation doesn't "overwhelm" relational information because each relational head contributes equally to each sensory head. This balance is essential for maintaining the generality of the architectural framework.

We will add further discussion to the revised paper explaining the intuitions behind the architectural design of DAT.

---

Thank you again for the discussion. We appreciate your thoughtful feedback.