<!-- Outline of final summarizing response at end of rebuttal period -->

Start with 10's review. Highlight praise.

Discuss 5's review. Highlight their concerns and how 

Discuss 3's review on GAT. Explain how their only comment was that our model is closely related to GAT. Explain how this is an inaccurate characterization: GAT does not explicitly represent flow of relational information between objects.

Discuss 3's review on Zhang et al. Start with summary of strengths. Explain how lack of novelty is only issue raised, and misses the main point of the paper. Also, explain how we discuss the papers mentioned from GNN literature in detail, and highlight key differences and areas of our main contributions.


---

[Summarize strengths, bolded, with references to reviewers that mentioned each strength.]

[Summarize concerns and criticisms of each review and our responses, together with reviewer acknowledgment if applicable.]


Review A (10)
- ...

Reviewer B (6)
- Though the review praises the architecture for being [well-motivated, and novel, as well as tackling an important problem], the review also raises some questions and concerns regarding the soundness of some aspects of the empirical evaluation, in particular regarding the role of symmetry, positional encoding, and some aspects of the presentation.
    - In our rebuttal, we aimed to address each point in turn by clarifying some details about the experimental set up as well as carrying out additional experiments. We were grateful that the reviewer responded to our rebuttal. The reviewer acknowledged that most of their concerns had been substantively addressed (and increased their score).
- One remaining point of concern that the reviewer raised in their final response concerns the role of different symbol assignment mechanisms and positional encoding.
    - In response, we carried out additional ablative experiments, which we hope address this final concern as well.


Reviewer C (3)
This review raised a single concern:
- A lack of novelty, citing a paper from the GNN literature by Zhang et al. (2023/24) that studies edge feature propagation in graph neural networks.
    - We refer to our response to the review, where we explain in detail the difference between the problems being tackled in our work and Zhang et al., and the difference in approach. To summarize, the main points are: 1) the problem of edge feature propagation in GNNs, while related in spirit, is distinct from the problem of relational representation learning in Transformers (we use the term "relation" to mean comparisons across different feature attributes, rather than edges or edge features in a graph or network); 2) The architectural proposals are distinct; 3) The application scope is distinct: i.e., sequence modeling within the Transformer framework (e.g., language modeling) vs graph processing within the GNN framework (e.g., molecular graph classification).
    - Nonetheless, we plan to include an expanded related work section in the final version of the paper which discusses various prior research efforts, including from the GNN community.
    - We note that novelty is praised as a *strength* of our work by Reviewer A and B [check].

Reviewer D (3)

- ["Novelty"]: The review claims that our work is "very similar to graph attention networks", which leads to the conclusion that it is not very novel.
    - The claim that relational attention (ours) is very similar to GAT is **factually inaccurate**. (Respectfully, we were confused by this assertion as we see almost no connection to GAT, besides the involvement of attention scores (which is the same as standard attention, and not unique to GAT).) We address this in detail in our response to the review, and hope that this cleared up any misunderstanding.
- "Experiments ": 
    - [Summarize the experimental evaluation, and scale of experiments]. [Explain how this goes far beyond the settings considered in the line of work on relational architectures that weaim to contribute to [Ref X-Y in paper].]
    - With the exception of reviewer D, there is consensus among all reviewers (A, B, C) on the strength and diversity of the experimental evaluation in the paper.

The reviews of C and D were short. We hope to have addressed the primary concerns in our rebuttal, attempting to engage the reviewers in discussion, though we unfortunately did not hear back. [Probably should rephrase]


---

# Summary of Reviews & Rebuttal Discussion

Dear all,

We would like to thank the reviewers for their reviews and thoughtful feedback, which has helped us to further improve the paper. This message summarizes the strengths and concerns raised in the reviews, our responses during the rebuttal, and the revisions made to improve the paper.

---

## Summary of Strengths

- Novel and well-motivated architectural proposal
    - "The authors take a well-motivated challenge (helping transformers work with relational reasoning) and define a natural extension to the transformer architecture that attack that challenge." (mxrQ)
    - "The proposed mechanism is novel, yet is a natural extension of the attention mechanisms within a standard transformer." (YUpf)
- Strong and diverse experimental evaluation
    - "The experimental data is impressive, especially because the architecture seems to work on a broad set of tasks. [...]  I appreciated the careful comparisons with baselines." (mxrQ)
    - "Empirically, the DAT appears to be more data efficient than the standard transformer, which is especially important in the case of language modeling." (YUpf)
    - "experiments are diverse in domains and supporting main claims" (qVFZ)
- Clear and effective exposition
    - "The paper is written clearly and easy to follow." (r4cd)
    - "The paper is well written and easy to follow" (qVFZ)

---

## Summary of Concerns and Responses

Below, we will summarize each review separately, highlighting the concerns raised and summarizing our responses and revisions. We refer to our individual responses to each reviewer for further details.

***Reviewer mxrQ***

We deeply appreciate reviewer mxrQ's enthusiasm for our work and their thoughtful, detailed feedback.

The primary concerns raised relate to exposition and presentation. We are especially grateful for the reviewer’s specific and constructive suggestions, which will significantly enhance the clarity and quality of the final version of the paper.

---

***Reviewer YUpf***

We'd like to thank reviewer YUpf for their detailed review and constructive feedback. We are grateful for the reviewer's response to our rebuttal, in which they stated that many of their **main concerns were "substantively addressed"**.

1. Concern: Impact of weight-tying of query/key maps in experiments of section 4.1
    - Response: We provided **additional ablative experiments** to address this specific question.
2. Concern: Role of positional information in symbol assignment
    - Response: **Additional experiments** using a symbol assignment mechanism without position-relative encoding (symbolic attention) confirmed our conclusions.
3. Concern: Clarification on terminology
    - Response: We point to our detailed response, where we clarify terminology and discuss the references mentioned by the reviewer.

---

The two remaining reviews were short, and we unfortunately did not receive a response to our rebuttal from the reviewers. We summarize our responses below, which we hope clarify and address their main concerns.

---

***Reviewer r4Cd***

1. Concern: "There is limited novelty. The relational attention sounds very similar to the graph attention network to me."
    - Response: **This characterization is inaccurate**. The only common feature between GAT and our proposed relational attention mechanism is that it involves computing attention scores, which is a feature shared with standard Transformer attention as well.
2. Concern: "The experiments are only performed on a small set of simpler tasks."
    - Response: **Our empirical evaluation includes complex real-world tasks, such as image recognition and language modeling.** Models with up to 1.3 billion parameters were evaluated to analyze scaling laws. These aspects were recognized as *strengths* by all other reviewers.

---

***Reviewer qVFZ***

1. Concern: The review questions the novelty of our work, citing the following paper from the graph neural network community "Learning Graph Representations Through Learning and Propagating Edge Features" by Zhang et al (2023) that studies edge feature propagation in GNNs.
    - Response: We point to our response to the review for a detailed discussion on the differences in scope, setting, and methodology between our work and Zhang et al. Here, we summarize the main points:
        1. **The two works study different problems:** relational reasoning in Transformers vs. edge feature propagation in GNNs.
        2. The **architectural proposals are distinct**.
        3. The **setting and application scope are distinct**: sequence modeling within the Transformer framework (e.g., language modeling) vs graph processing within the GNN framework (e.g., molecular graph classification).

---

We hope our responses and revisions effectively address the concerns raised. 

Best Regards,

Authors