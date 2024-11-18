We'd like to sincerely thank you for your deep engagement with our work, and your thorough evaluation and valueable feedback! We are encouraged that you found the problem to be "well-motivated", the architectural proposal to be a "natural extension to the transformer architecture", and the experimental evaluation to be "impressive", covering a "broad set of tasks" with "careful comparisons with baselines".

We appreciate your many useful and thoughtful suggestions around presentation. We will carefully consider all of them.

Below, we respond to some of your comments.

---

> The experimental data is impressive, especially because the architecture seems to work on a broad set of tasks. It's interesting that even image recognition improves.

This was interesting to us as well! As discussed in the paper, in the work on relational inductive biases that most influenced ours, empirical evaluation was mostly limited to synthetic benchmarks, similar to the relational games benchmark in Section 4.1 [see e.g., references 15-22 in the paper]. So it was an open question as to whether these ideas can yield improvements in complex (and messy) real-world tasks, like image recognition and language modeling. This was a big part of the motivation for our work, and the decision to build on the powerful Transformer framework. We were certainly encouraged to see that these relational mechanisms yield meaningful performance improvements across a range of complex tasks, while maintaining the generality of the Transformer architecture!

---

> Crucially, the new architecture decouples strength of attention from strength of relation (this distinguishes it from an earlier proposal known as relational cross attention).

Yes! As you may have seen, we have a section in the appendix where we discuss the relationship between relational attention and RCA, and present some exploratory experiments considering the performance a DAT variant with RCA. Interestingly, we find that although RCA performs comparably on the synthetic relational games experiments, it is significantly worse at language modeling: an RCA-based DAT performs seemingly identically to standard Transformers, and loses the improvement due to relational attention.

---

> I believe there are some opportunities for improving the exposition of this paper. To begin with, "sensory" doesn't seem like the right metaphor. I realize the cognitive science origin, but I also think it's worth being careful with brain metaphors. I wonder if it would be better to talk in terms of "unary" vs. "binary" attention heads, or "first-order" vs. "relational," perhaps.

These are interesting suggestions. We take your point about the accuracy of the term "sensory". The term sensory refers to the fact that the values contain features of the objects. Although we like the cognitive science reference, we agree that such brain metaphors can sometimes be misleading. We will carefully consider your suggestions.

> The first paragraph (and maybe much of the second) of the introduction seem unnecessary, and it might be possible to cut them entirely.

It's always helpful to get feedback on exposition and presentation. We will try to make the presentation more succinct.

> I found the first few explanations of the architecture confusing, and didn't really understand the "type" of r or symbols until I got to the explicit formulas. I wonder if it's worth making this a little more precise earlier.

This is useful feedback! This was a concern for us while writing as well, and its useful to have this confirmed. We will aim to revise accordingly.

> The theorem in 2.4 gets very little play, and I'm not sure how important it is. I'd recommend either relegating this entirely to the appendix, or spending a bit more time explaining why it matters here. (One issue is that plain-vanilla transformers are computationally very powerful already, so it's not clear what this theorem adds.)

Thank you for this feedback on exposition.
Our intention for placing the theorem in the main text of the paper is to try to give some intuition about the class of functions that relational attention computes, which we thought might be helpful for readers who like more formal statements. In particular, the theorem aims to make clear how the attention criterion is decoupled from the relation being modeled.

But given your feedback, we will carefully think about the presentation of this Theorem, and perhaps either expand on its significance or move it to the appendix.

---

> Figure 5 is potentially interesting, but I wonder if the story could be illustrated better by picking one layer, and showing attention for all the "normal" heads vs. the "relation" heads. I also wonder if there is any "low-hanging fruit" for other visualizations. For example, for training when the relation matrices are not constrained to be symmetric, do they ever end up learning to be near-symmetric? That said, this is a long paper already, and the authors explicitly mention interpretability as future work, so this is certainly an optional change!

We'd certainly love to explore interpretability further, and your suggestions make sense! What we'd ultimately like to do is to compare the structure between three things: the attention scores in standard attention, the attention scores in relational attention, and the relations in relational attention. There are many interesting questions here, and we'd like to take the time to address them rigorously and quantitatively. For the current paper, however, we will also think about other possibilities to improve Figure 5 and provide a visualization of the learned relations.

> As just mentioned—and I can't fault the authors for this—this paper has a huge amount of material, mostly in the appendices. All of that is good and necessary, but it's easy to miss details when reviewing, which is why I've put a relatively low confidence score in my review.

Totally understandable! Our goal was to make the main body of the paper as self-contained as possible, while including a more thorough presentation of the relevant details in the appendix. We think we will be able to further improve the presentation with your helpful feedback.

Please let us know if there are any details that we can help to clarify for you.

---

> If you cut some of the text as suggested above, you might have more room for future work. Do you have thoughts about how this might apply to other architectures, such as graph neural nets?

The case of graph neural networks is interesting, and deserves an in-depth discussion. A key aspect in the case of graph neural networks is the distinction between edges on the graph (and possible features of these edges), and the relations between nodes (the terminology can be overloaded here sometimes). This can enable some interesting interaction between these two aspects, since in the standard message-passing paradigm for GNNs, the role of the edges is to control the flow of node information. Whereas the *DAT* considers fixed "graphs" (i.e., either fully-connected or causal), a GNN-variant of our proposal could enable some interesting interaction between the *direction of information propagation* (i.e., edges) and the *relational content* of the information being propagated.

Another interesting question is whether it is possible to integrate an analogous notion of relational processing in recurrent sequence models such as the recent SSM class of models, or if such relational processing is unique to attentional models like Transformers that have direct access to the entire context.