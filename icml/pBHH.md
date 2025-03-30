# Review
---

Summary:
The authors describe a neural architecture (DAT) in which relational information is a first-class object, and via a series of experiments show that this architecture offers genuine empirical benefits.

Claims And Evidence:
The claims seem solid. The question with this type of paper is often less about the solidity of the claims, however, but more about their significance. To be concrete, why do we need yet another transformer-like architecture? In this case, I think the new architecture meets the bar of significance for publication. First, it can be seen as helping make explicit the type of computation that "classic" transformers are learning implicitly. Second, it appears that for certain tasks DAT actually outperforms transformers. Third, it opens up interesting possibilities for interpretability work, potentially making neural systems more transparent.

Methods And Evaluation Criteria:
The paper has a good balance of methods, analyzing the architecture theoretically as well as performing experiments.

Theoretical Claims:
The theorem in Appendix A seems correct, although the Debreu representation theorem was new to me. Once I knew what that was, the result seemed correct. I'd recommend that the authors provide a paragraph outlining the result at a high level, which I read as: selection can be modeled with a preference ordering (with some mild conditions); the Debreu representation theorem allows us to think of computing such a preference ordering as computing a certain continuous function; we then just use standard results saying neural nets can approximate continuous functions. However, I'd note to the authors: if this outline is NOT what the proof is actually saying, then please clarify!

Experimental Designs Or Analyses:
The designs seem solid, and I appreciate that there are both language and vision tasks, which strengthens the overall claim.

Supplementary Material:
n/a

Relation To Broader Scientific Literature:
The question of whether and how transformers learn relationships is definitely central, so this seems squarely in the mainstream. See next section for more detail.

Essential References Not Discussed:
I think the related work cited is fine. If the authors want to expand a bit, there are two areas where I don't see citations; however, they may not really be essential. One is recent work on how transfomers do seem to represent relations, e.g. "How do Language Models Bind Entities in Context?" (Feng & Steinhardt) and papers that cite it. Another is theoretical ways that have been proposed for representing relations (vector symbolic architecture, etc.)

Other Strengths And Weaknesses:
Figure 8 in the appendix is intriguing, and interpretability work would probably be an entire new paper. I do want to flag one thing about the claim that "Relational attention in DAT language models encodes human-interpretable semantic relations" from the caption: in many cases, classic transformers, too, encode human-interpretable semantic relations. (And here the relation isn't particularly striking: it just seems like it's picking out generally related words?) The fact that one can find a single example of an attention head that is interpretable isn't particularly interesting or useful in itself. I wonder if there's an example of an attention head which finds some kind of relation that is not normally seen in a classic transformer?

Other Comments Or Suggestions:
I'm not fond of the term "sensory" here, because it seems actively misleading. There's already a common metaphor for deep networks where people talk about the early layers as doing sensory processing, and the final layers as being analogous to motor neurons.

I'd recommend instead calling this something like "first-order" vs. "relational" information.

Questions For Authors:
none, except for the note about confirming my read of the proof, mentioned above.

Code Of Conduct: Affirmed.
Overall Recommendation: 4: Accept

---

# Response

---

Thank you for your detailed and thoughtful review. We appreciate your positive feedback about the significance, methodological soundness, and strength of empirical evaluation. Below, we hope to address the main concerns you raised in turn.

**D1: Interpretability of Learned Relations**

> Figure 8 in the appendix is intriguing, and interpretability work would probably be an entire new paper. I do want to flag one thing about the claim that "Relational attention in DAT language models encodes human-interpretable semantic relations" from the caption: in many cases, classic transformers, too, encode human-interpretable semantic relations. (And here the relation isn't particularly striking: it just seems like it's picking out generally related words?) The fact that one can find a single example of an attention head that is interpretable isn't particularly interesting or useful in itself. I wonder if there's an example of an attention head which finds some kind of relation that is not normally seen in a classic transformer?

Thank you for your thoughtful comments regarding the interpretability of learned relations. This is indeed an interesting and important question that we’ve begun to explore further. We are glad you found the preliminary exploration in Figure 8 intriguing. As you rightly point out, the relations observed in Figure 8 appear to encode relatively simple semantic similarity between words/tokens, which has also been observed in the attention scores of standard Transformer attention. However, it is important to note that these relations serve a different functional role here: while the attention scores $\alpha_{ij}$ in standard Transformers encode a *selection criterion* controlling *where* information is routed, in *DAT*, the relation vectors $\boldsymbol{r}_{ij}$ serve as the *values* being transmitted between tokens, controlling *what* information is routed (with an independent set of attention scores controlling the selection criterion).

We agree with you that finding a single interpretable relational attention head may not be very revealing on its own, and we would be excited to investigate  whether relational heads in *DAT* might encode novel relationships that are not typically seen in classic Transformers. In particular, it would be interesting to understand how these relations form computational circuits that are unique to the DAT architecture---that is, characterize specific computational circuits that use relational heads in unique ways to carry out a specific computation, understanding their functional role on a deeper level beyond just the fact that the relations themselves appear human-interpretable.

For now, this interpretability work is outside the scope of this paper, but we intend to explore it further in future work. One initial step we’ve taken is developing an interactive application that allows users to load pre-trained DAT models and visualize relational representations at various layers on their own inputs. We plan to include a link to this tool in the final, de-anonymized version of the paper.


---

**D2: Question on proof of Representational Capacity Theorem.**

> The theorem in Appendix A seems correct, although the Debreu representation theorem was new to me. Once I knew what that was, the result seemed correct. I'd recommend that the authors provide a paragraph outlining the result at a high level

Thank you for question and the suggestion to provide further discussion on the Debreau representation theorem. We would be happy to add further discussion, providing a high-level overview of the Debreau representation theorem and the related literature.

Yes, your interpretation of the result and its proof are correct. The Debreau representation theorem, due to the economics literature, identifies preference relations on a topological space or metric space with a continuous "utility" function, assuming certain continuity properties on the preference relation with respect to the underlying topology. For us, the key is to extend this to a *family* of query-dependent preference relations in order to specify the attention mechanism. That is, each query is associated with an ordered space, and we require continuity of the family of preference relations with respect to both queries and keys, which we formulate as query-continuity and key-continuity, respectively. From there, the result follows by the approximation properties of inner products of MLPs.


**D3: Further discussion of related work**

> If the authors want to expand a bit, there are two areas where I don't see citations; however, they may not really be essential. One is recent work on how transfomers do seem to represent relations, e.g. "How do Language Models Bind Entities in Context?" (Feng & Steinhardt) and papers that cite it. Another is theoretical ways that have been proposed for representing relations (vector symbolic architecture, etc.)

Thank you for these suggestions. We will incorporate them into our discussion of related work.