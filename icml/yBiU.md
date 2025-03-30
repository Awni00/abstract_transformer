# Review

Summary:
This paper presents the Dual Attention Transformer (DAT), an extension of the Transformer architecture that introduces a relational attention mechanism alongside the standard self-attention mechanism. The key idea is to explicitly represent and process relational information by replacing the standard value aggregation in self-attention with a weighted combination of a relation vector computed for each pair of objects and a symbol vector.

Claims And Evidence:
The paper argues that standard Transformers primarily process sensory information and struggle with relational reasoning due to the entanglement of sensory and relational information. The proposed DAT disentangles these components, leading to improved performance. The empirical results largely support these claims.

Methods And Evaluation Criteria:
The authors employ standard benchmarks to assess the effectiveness of DAT, comparing it against standard Transformer models across various tasks.

Theoretical Claims:
This paper primarily focuses on the empirical aspects.

Experimental Designs Or Analyses:
It might be more convincing if more baseline comparisons with graph-based models and message-passing networks are included, given the conceptual similarities.

Supplementary Material:
I reviewed implementation details and experiment details in supplementary material.

Relation To Broader Scientific Literature:
The paper presents contribution to the development of relational reasoning in Transformer architectures, with potential implications across multiple domains, including language processing and vision. By introducing an explicit mechanism for processing relational information, the proposed approach highlights the importance of integrating structured reasoning capabilities into deep learning models.

Essential References Not Discussed:
It might be helpful if more key works in graph neural networks that have incorporated relational attention mechanisms were discussed and compared.

Other Strengths And Weaknesses:
The strengths and weaknesses have been addressed in earlier sections, and no additional ones require emphasis here.

Other Comments Or Suggestions:
The comments and suggestions have been addressed in earlier sections, and no additional ones require emphasis here.

Questions For Authors:
How does DAT perform on more complex benchmarks beyond the current experimental setup?

Code Of Conduct: Affirmed.
Overall Recommendation: 3: Weak accept (i.e., leaning towards accept, but could also be rejected)

---

# Response

Thank you for your thoughtful review and positive feedback on the overall contribution of our work, especially for highlighting the empirical support for our claims, the contribution to relational reasoning in Transformer architectures, and the importance of structured reasoning in deep learning models. Below, we hope to address the key comments and concerns you raised.

---

**C1: Comparison with Baselines**

Thank you for your suggestion regarding additional baseline comparisons.

We would like to highlight **Appendix C** (specifically Section C.1) and **Appendix D**, where we compare our proposed method to several prior works on relational learning. In particular, we compare our model to PrediNet (Shanahan et al. 2020), CoRelNet (Kerg et al. 2022), and the Abstractor (Altabaa et al. 2024), positioning our work within this ongoing line of research on relational reasoning and extending these prior efforts.

One way to describe the architectures in these prior works is that they incorporate *"substractive"*, inductive biases that constrain the types of representations the model can compute (see [Ref 24] on the "Relational Bottleneck" for more on this). Due to these strict inductive biases, these architectures are narrow in domain and have mostly been applied to simple synthetic tasks, like the Relational Games benchmark introduced by Shanahan et al. (2020). By contrast, our approach in developing the *Dual Attention Transformer* architecture is *"additive"*, in the sense that it incorporates new explicit new relational processing capabilities without constraining existing components of the Transformer architecture, allowing the model to learn to select between the different computational mechanisms available to it based on the task or context, as well as compose them to create flexible and expressive computational circuits.

Despite these differences, it is nonetheless useful to compare *DAT* against those architectures on controlled synthetic benchmarks to explore the trade-offs of strong inductive biases and evaluate *DAT* in comparison to alternative approaches to relational learning. This was carried out and discussed in Appendix C.1.

Initially, due to space constraints, we deferred this discussion to the appendix. However, with the additional page allowance, we plan to incorporate this more detailed comparison and discussion of related works into the main body of the paper. Please also see our response numbered **A1** to reviewer oPuj, where we discuss a similar question.

---

**C2: Conceptual similarities to message-passing networks**

As mentioned, we view the aforementioned line of work on relational learning [Ref. 19, 20, 21, 22, 23] as the most closely related literature to our work. However, we agree that there are some conceptual similarities between our *DAT* architecture and message-passing networks, which is shared by the Transformer framework more broadly. In particular, like standard Transformers, the *DAT* architecture can be describe in the language of the message-passing framework. In message-passing terminology, the messages exchanged in standard Transformer attention encode first-order sensory features of the sender, while in *relational attention*, the messages encode relational features between the sender and the receiver.

We will additionally incorporate a discussion on the conceptual connections between the *DAT* architecture and message-passing networks into the paper.