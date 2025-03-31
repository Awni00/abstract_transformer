# Review
Summary:
The authors propose a parameter-efficient variant of the self-attention mechanism in transformers called the Dual-Attention Transformer (DAT), which explicitly routes both sensory information (about individual tokens) and relational information (about relationships between pairs of tokens). The key differences between relational attention and standard attention are that 1) instead of routing a value projection from the query, a vector of relational similarities is constructed (by concatenating the dot products of multiple learnable relation projections between query and key), and 2) A symbol vector (from a learned codebook), retrieved for each key, is added to the corresponding relation vector, and the result is then routed with the usual attention weight.

The authors then incorporate both standard and relational attention heads in a multi-headed attention framework to construct DAT models and demonstrate their effectiveness across various domains - from explicitly relational tasks in the RelationsGame to language modeling, mathematical problem-solving, and image recognition tasks.

Claims And Evidence:
The paper makes two primary claims that are well-supported by the evidence:

DAT outperforms standard transformers with multi-headed attention - this is demonstrated convincingly across multiple domains and tasks
The relational attention mechanism better routes relational information than standard self-attention - this is most clearly shown on explicitly relational tasks from the RelationsGame
Methods And Evaluation Criteria:
The methods and evaluation criteria are appropriate for the demonstrations. The authors evaluate on a diverse set of tasks spanning different domains (visual reasoning, symbolic math problems, language modeling, image classification) and compare against comparable transformer baselines (or relational baselines) while controlling for parameter count.

Theoretical Claims:
A theoretical claim is made about the class of functions that can be computed by relational attention. A proof is provided in the appendices and was not checked. The claim is not particularly strong and the informal proof in the main text seems sufficient.

Experimental Designs Or Analyses:
The experiments and ablations are thorough, well-reported, and compelling. The learning curves on across most tasks show clear data efficiency advantages of DAT over standard transformers. Furthermore, the additional results in the supplementary material are comprenehsive.

Supplementary Material:
The supplementary materials are extensive. I primarily reviewed the relational baselines in figure 6; for most claims the results in the main body was sufficient.

Relation To Broader Scientific Literature:
The paper's findings are moderately significant and relevant to current research on improving transformer architectures. The authors position their work appropriately within the literature on relational inductive biases and transformer models.

One potential limitation not fully addressed is whether the performance gains from RelationalAttention would still be significant at very large scales, or if they might be incompatible with optimization tricks for self-attention. However, the consistent improvements across the scales explored are promising and warrant further exploration.

Essential References Not Discussed:
None of which I am aware

Other Strengths And Weaknesses:
None of note

Other Comments Or Suggestions:
The ICML Style guide requires citations within the text to include authors' last names and year, which this paper doesn't follow
Line 161 contains an unnecessary sentence fragment "this adds structures..."
Questions For Authors:
A potential downside of DAT even within the experiments carried out is the possibility that the additional computations required for RelationalAttention outweigh the gains from parameter efficiency. Though differences are likely marginal, was this something you investigated?
For very large transformers trained on extensive data, standard self-attention may eventually capture relational information in complex tasks while retaining greater efficacy for other common computations - potentially narrowing the performance gap at increasing scale. While large-scale demonstrations aren't necessary for this paper, do you have intuitions about how the advantages of DAT might scale?
Code Of Conduct: Affirmed.
Overall Recommendation: 4: Accept

---

# Response

Thank you for your detailed and thoughtful review. We appreciate your positive assessment of our work and are encouraged by your recognition of its methodological soundness, strong empirical results, and relevance to the literature on relational inductive biases and Transformer-based architectures. Below, we will aim to respond to the key comments and concerns you raised.

---

**B1: Computational efficiency**

> A potential downside of DAT even within the experiments carried out is the possibility that the additional computations required for RelationalAttention outweigh the gains from parameter efficiency. Though differences are likely marginal, was this something you investigated?

Thank you for raising this important point. Our experiments were designed to carefully control for model size (i.e., parameter count) when comparing the *DAT* architecture to baselines. In particular, the parameter count is slightly *smaller* for the *DAT* model compared to the baselines. While we did not explicitly measure computational cost in terms of FLOPS, we expect the differences to be marginal given the parameter count differences.

From a practical standpoint, we believe that the more important factor in computational efficiency is the availability of optimized GPU kernels, such as FlashAttention. This gap could perhaps be bridged given interest from the MLSys community to develop optimized kernels for relational attention, but for now this would be an obstacle to scaling *DAT*-style architectures in a cost-effective way.

---

**B2: Scaling to Larger Models**

> One potential limitation not fully addressed is whether the performance gains from RelationalAttention would still be significant at very large scales, or if they might be incompatible with optimization tricks for self-attention. However, the consistent improvements across the scales explored are promising and warrant further exploration.

> While large-scale demonstrations aren't necessary for this paper, do you have intuitions about how the advantages of DAT might scale?

Our intuition, guided by our scaling results up to ~1B-parameter scales, is that the explicit relational processing capabilities of *DAT* will continue to provide significant benefits at even larger scales. Our hypothesis is that relational processing is a key computational capability which is useful in many domains and at several levels of abstraction---having this be explicitly supported by a model architecture can enable more efficient learning and greater generalization capabilities.

Of course, confirming this hypothesis requires empirical validation, which we currently do not have the resources to do at our academic institution. As you rightly mentioned, and as discussed above, scaling further introduces new challenges and the availability of various optimization tricks becomes an important consideration. We hope future work will explore this further.

---

Thank you again for your thoughtful review.