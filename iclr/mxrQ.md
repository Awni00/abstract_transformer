[Thank for positive feedback, deep engagement, and valueable suggestions.]

[We are excited and encouraged by the positive feedback, and are grateful for the many useful and thoughtful suggestions.]

---
[Begin with response / discussion of strengths]

> The experimental data is impressive, especially because the architecture seems to work on a broad set of tasks. It's interesting that even image recognition improves.

This was interesting to us as well! As discussed in the paper, in the work on relational inductive biases that most influenced ours, empirical success was mostly limited to synthetic benchmarks, similar to the relational games benchmark in Section 4.1 [see e.g., references 15-22 in the paper]. So it was an open question as to whether these ideas can yield improvements in complex (and messy) real-world tasks, like image recognition and language modeling. This was a big part of the motivation for our work, and the decision to build on the powerful Transformer framework. We were certainly encouraged to see that these relational mechanisms yield meaningful performance improvements across a range of complex tasks, while maintaining the generality of the Transformer architecture!

> Crucially, the new architecture decouples strength of attention from strength of relation (this distinguishes it from an earlier proposal known as relational cross attention).

Yes! As you may have seen, we have a section in the appendix where we discuss the relationship between relational attention and RCA, and present some exploratory experiments considering the performance a DAT variant with RCA. Interestingly, we find that although RCA performs comparably on the synthetic relational games experiments, it is significantly worse language modeling (now performs seemingly identically to standard Transformers, and losing the improvement due to relational attention).

---

> I believe there are some opportunities for improving the exposition of this paper. To begin with, "sensory" doesn't seem like the right metaphor. I realize the cognitive science origin, but I also think it's worth being careful with brain metaphors. I wonder if it would be better to talk in terms of "unary" vs. "binary" attention heads, or "first-order" vs. "relational," perhaps.

These are interesting suggestions. We take your point about the accuracy of the "sensory" metaphor. Although we like the cognitive science analogy, we completely agree that such brain metaphors can sometimes give the wrong idea. We will carefully consider your suggestions.

> The first paragraph (and maybe much of the second) of the introduction seem unnecessary, and it might be possible to cut them entirely.

It's always helpful to get feedback on exposition and presentation; thank you!

> I found the first few explanations of the architecture confusing, and didn't really understand the "type" of r or symbols until I got to the explicit formulas. I wonder if it's worth making this a little more precise earlier.

This is useful feedback! This was a concern for us while writing as well, and its useful to have this confirmed. We will aim to revise accordingly.

> The theorem in 2.4 gets very little play, and I'm not sure how important it is. I'd recommend either relegating this entirely to the appendix, or spending a bit more time explaining why it matters here. (One issue is that plain-vanilla transformers are computationally very powerful already, so it's not clear what this theorem adds.)

---

> Figure 5 is potentially interesting, but I wonder if the story could be illustrated better by picking one layer, and showing attention for all the "normal" heads vs. the "relation" heads. I also wonder if there is any "low-hanging fruit" for other visualizations. For example, for training when the relation matrices are not constrained to be symmetric, do they ever end up learning to be near-symmetric? That said, this is a long paper already, and the authors explicitly mention interpretability as future work, so this is certainly an optional change!

We'd certainly love to explore interpretability further, and your suggestions make sense! [ Maybe try some easy things out, give some teasers, and say that we will aim to explore further...]

> As just mentioned—and I can't fault the authors for this—this paper has a huge amount of material, mostly in the appendices. All of that is good and necessary, but it's easy to miss details when reviewing, which is why I've put a relatively low confidence score in my review.

---

> If you cut some of the text as suggested above, you might have more room for future work. Do you have thoughts about how this might apply to other architectures, such as graph neural nets?

The case of graph neural networks is interesting, and deserves an in-depth discussion. One interesting aspect about graph neural networks with respect relational representations is how graph-edges would interact with relation between node embeddings.

[Expand...]