> The idea does not seem very novel or original. There are many attempts to integrate relational information into the attention mechanism in the Graph Neural Network community, with the closest one I can find being "Learning Graph Representations Through Learning and Propagating Edge Features" (https://ieeexplore.ieee.org/document/10004977). Specifically, Eq. 2 directly gives the general form of the proposed relational attention. This previous work goes on with slightly different parametrization of f and g, i.e. concatenation instead of dot product etc, but has an overall very similar central idea. This paper should at least cites this line of work and compare against them as baselines.

Thank you for your review, and for your positive comments regarding the presentation of the paper and the strength and diversity of our experiments with respect to the paper's main claims.

We appreciate the reference to Zhang et al.'s work, which we were not aware of. Our work indeed shares certain similarities with their framework in the sense that both approaches seek to integrate relational representations into neural models. However, our approach is distinct in both scope and application: ***while Zhang et al. address the propagation of edge features in Graph Neural Networks, our focus is on adapting relational representation learning specifically within Transformers for sequence modeling tasks.***

In response to your suggestion, we will expand the related work section in the revised version of the paper, and incorporate a discussion on the similarities and differences to the work of Zhang et al., as well as other relevant work from the GNN community.

Below, we hope to address your concern around the relationship to previous work in the GNN literature in detail.

## Transformers vs GNNs

**Our work studies relational representation learning *in Transformers*, while the work you mention studies integrating relational information (edge features) specifically *within GNNs*.** Although Transformers and GNNs can be linked (by viewing Transformers as GNN-variants operating on a fully connected graph), they are ***distinct architectural paradigms that tackle a different class of tasks and have various differing considerations***. Moreover, the role of relational information in each paradigm is different in important ways.

In particular, GNNs operate over *graphs*, where the primary inputs include edges, edge features, and node features, making GNNs highly suitable for graph-structured data such as social networks, molecular graphs (e.g., ZINC), and macromolecular structures (e.g., ENZYMES, PROTEINS datasets). By contrast, Transformers are designed as *sequence models* and are typically applied to tasks such as language modeling, machine translation, and other forms of sequential data processing.

The work by Zhang et al. specifically focuses on propagating edge features within a GNN through a message-passing paradigm. Here, edge features are a core part of the input (along with graph edges and node features) and are propagated along graph edges during the message-passing operations. For example, Zhang et al. applies this to molecular graphs, where the edge features are bond types. This is structurally and conceptually distinct from the mechanisms we develop within the Transformer framework.

**A key contribution of our work is to propose computational mechanisms for relational processing *within the Transformer framework*, and demonstrating empirically that this approach offers significant performance improvements in terms both data efficiency and parameter efficiency across a diverse range of sequence-based tasks.** We view this as ***distinct from research efforts in the GNN community*** to integrate edge features into graph-processing through message-passing operations along graphs.

<!-- Below, we will discuss the analogy between the *relational attention* mechanism proposed in our work and the mechanism proposed in the paper you mention.  -->

<!-- keep this or not; is it too subtle? -->


## Key differences to the approach of Zhang et al. (2024)

> Specifically, Eq. 2 directly gives the general form of the proposed relational attention

While our work and Zhang et al. (2024) both address integrating relational information in neural models, our approach differs significantly in its architectural framework, task setting, and specific mechanisms.

Equation 2 in Zhang et al. serves as a generic formulation of propagating edge features within the message-passing framework of GNNs. This is a useful expository tool to motivate their architecture, but it is not a concrete architectural proposal. Rather, it outlines a broad class of possible models, similar to how the message-passing framework in GNNs is formulated in general terms. In fact, in Equation 1 of our introduction, we write a similar equation to motivate and formulate the problem, but again, neither is intended as as a full architectural proposal.

This distinction is akin to how the message-passing framework in GNNs captures a broad class of models, but by itself does not define a specific architecture (e.g., for a more in-depth discussion, see Chapter 5 of W. L. Hamilton’s “Graph Representation Learning,” which provides a nice exposition on neural message-passing in GNNs).

> This previous work goes on with slightly different parametrization of f and g, i.e. concatenation instead of dot product etc,

These details are crucial, and are what forms the primary contribution of both our work and Zhang et al's, within the respective frameworks of Transformers and GNNs. In particular, their approach is designed for propagation of edge features on graph inputs, whereas our approach is an extension of the Transformer framework with relational processing mechanisms for sequence data.

The proposal of Zhang et al. is an operation which processes a graph consisting of a collection of nodes $\{x_i\}_{i \in \mathcal{N}}$, edges $\mathcal{E} \subset \mathcal{N} \times \mathcal{N}$, and *edge features* $\{e_{uv} : u,v \in \mathcal{E} \}$. They propose updating the edge features by applying a linear map to the concatenation of the initial edge features and the pair of node features:
$$e_{uv}' = V \cdot \mathrm{concat}(x_u, e_{uv}, x_v)$$
This is then aggregated by an attention mechanism (though not the standard dot-product attention typically used in Transformers).

In contrast, our relational attention mechanism is defined as:
$$\mathrm{RelAttn}(x, (y_1, ..., y_n)) = \sum_{i} \alpha_{i}(x, \bm{y}) (W_r r(x, y_i) + W_s s_i),$$
where $\alpha_i(x, \bm{y})$ are dot-product attention scores, $r(x, y_i) \in \mathbb{R}^{d_r}$ is a relation function parameterized as a series of inner product comparisons under different learned feature projections, and $s_i$ is a vector which acts as a pointer or reference to the object $y_i$ the relation is with.

We highlight some key key differences between the two proposals:

- **Task settings:** Zhang et al.'s work is tailored for graph-based tasks in GNNs, where the input consists of nodes, edges, and edge features. In contrast, our approach is designed for sequence modeling with Transformers, where the input is a sequence and does not involve graph structures or edge features. For example, the tasks Zhang et al. tackle include modeling of molecular graphs (ZINC), macromolecular graphs (PROTEINS, ENZYMES), and synthetic benchmarks generated by stochastic block models (PATTERN, CLUSTER), formulated as node classification or graph classification. In contrast, we consider sequence modeling tasks, such as autoregressive language modeling, sequence-to-sequence symbolic processing, and visual processing (by treating image patches as a sequence).
- **Relation modeling:** The way that the relations are modeled differs fundamentally. We model relations as inner products of learned feature projections. In contrast, Zhang et al. models updated edge features as a linear map applied to the node pair of node features and the initial edge features (which is an input to the model).
- **Symbol assignment mechanisms:** The use of symbol assignment mechanisms, serving as pointers to objects in relational processing, is unique to our model and unique to the sequence modeling setting (as opposed graph processing)
- **Dual attention mechanisms:** Our proposal includes *dual attention*, a variant of multi-head attention with both sensory and relational processing mechanisms, is a novel contribution of our work. It is also specific to the Transformer framework.
- **Dual Attention Transformer:** The proposal of a corresponding extension to the transformer framework, the *Dual Attention Transformer (DAT)*, is a novel contribution of our work.

> This paper should at least cites this line of work and compare against them as baselines.

Thank you for pointing out this related line of work. We agree that citing and discussing these papers will strengthen our discussion of related models, and we will add an expanded related work section to do so.

However, a direct comparison with these models as baselines would not be appropriate, given that the primary focus of this work is on sequence modeling and the Transformer framework, while the mentioned works are centered on graph neural networks (GNNs). GNN architectures are designed for learning over graph-structured data, and thus do not naturally extend to sequence modeling tasks without substantial architectural modifications that go beyond the scope of this work.


## Related work

***In the revised version of the paper, we will include an expanded related work section, which will discuss the relation between our work and prior work in-detail***. This will include the Zhang et al. paper you mentioned, other work in the GNN literature, and an expanded discussion on the line of work we were influenced by.

We briefly note that our work is most influenced by a line of work on relational architectures, which falls outside the GNN literature. Notably, this includes including RelationNet (Santoro et al), PrediNet (Shanahan et al), and Abstractor (Altabaa et al.). We will expand the discussion on these works as well.


[Is this repetitive? Mentioned something like this above...]
## Contributions of this work

Finally, we would like to remind the reviewer of our main contributions in this work:
- The proposal of a neural mechanism for routing and processing relational information *within the Transformer framework*. The proposed *relational attention* mechanism is based on an attentional operation that selectively retrieves relational information from the context, tagging it with symbolic identifiers.
- The proposal of the *dual attention* mechanism, a variant of multi-head attention with *two distinct types of attention heads*, enabling routing and processing of both sensory information and relational information.
- The proposal of a corresponding ***extension to the Transformer framework*** called the *Dual Attention Transformer (DAT)*, which integrates sensory and relational information in a unified architecture. The strength of this framework is that it is as general as the Transformer framework, while yielding benefits in flexibility, data efficiency, and parameter efficiency. In particular, it supports all architecture variants of the standard Transformer (e.g., encoder-only, decoder-only, encoder-decoder, ViT, etc.), and can be applied across a diverse range of tasks and data modalities.
- We evaluate the proposed *DAT* architecture on a diverse set of tasks ranging from synthetic relational benchmarks to complex real-world tasks such as language modeling and visual processing, demonstrating notable improvements across all tasks. This in particular includes an analysis of scaling laws, which demonstrate greater data efficiency and parameter efficiency at large scales.

---

> Questions:
> Are there experiments with the learned Symbolic Attention?

Yes, the language modeling experiments of section 4.4 use Symbolic Attention. Interpreting symbolic attention as a learned differentiable equivalence class over embeddings, we conjecture the symbolic attention learns to represent semantic structures, perhaps analogous to synsets. We are excited to explore this further in future work as part of a broader mechanistic interpretability investigation.

---

## Dump of previously-written responses that were trimmed or removed

- [Distinguish between Transformers and GNNs. Related but different in significant ways. The methods he mentions are GNNs applied to graph-structured data such as social networks, molecular graphs (ZINC), macromolecule graphs (ENZYMES, PROTEINS), etc. Within the Transformer framework, the tasks of interest are very different, typically focusing on discrete sequence data such as text.] [These vastly different settings, so we respectfully disagree that the line of work in the GNN literature that you mention render our work unoriginal.]
- [The GNN message passing framework is so general that it vacuously captures many architectures as a special case. (Maybe vacuous is too strong, so rephrase, but the point is that you lose resolution of important details when you say that two architectures are the same simply because they fall into the message-passing framework)]
- [We appreciate the neural message-passing framework as a conceptual framework for thinking about different architectures for models operating over collections of objects. Indeed, we use this conceptual]
- [Emphasize difference in proposed architecture between our work and the cited paper, with respect to intended use and intended tasks. Although the framework and proposal are analogous, they tackle quite different problems and result in quite different final solutions.].
- [Emphasize novelty *within* the Transformer literature, and the significance of that as a contribution.]
- [Mention our motivating literature (ESBN, CoRelNet, PrediNet, Abstractor)]
- [Promise an expanded related work section which discusses and cites the relevant research efforts in the GNN community.]

- Somehow make the point that if you apply the rule that anything that is a special case of Eq (2) in the cited paper is not original, then nothing will be original.

- Our work is within the Transformer framework, whereas the work you cite 
- The GNN literature has considered different 


- Equation (2) is not the definition of an architecture, it is the general form of . [We note that the paper you mention is not the origin of this general paradigm (soften in case this is there paper; maybe start discussion with mention of generality of message-passing paradigm)]. See e.g.,  To say that our architecture is a special case of Equation (2) is vacuous: every variant of a Transformer architecture and every GNN is a special case of Equation (2).
- Although GNN and Transformers are related

[Maybe this could be the structure of the response]

- Summarize their criticism
- Discuss the general paradigm of message-passing networks. Discuss how it relates to modeling edge features.
- Highlight similarities and difference in setting, architecture, tasks, etc. (start with similarities, and maybe appease them about the paper. spend more time explaining contributions and differences in *setting* rather than difference in architecture).
    - Form of $g$ in paper is a linear map applied to the concatenation of object embedding pair (together with previous relation). For us, relations are modeled as inner products which compute *explicit comparisons* between the two objects features. [In their comment, they gloss over the "specific choices of $g$ and $f$; but those are crucial and are what forms the actual contributions of different work; the message-passing paradigm captures a huge range of possible model architectures, and the contribution of individual papers is to explore that space for interesting architectures.]
- Discuss work that motivated our paper (e.g., ESBN, CoRelNet, PrediNet, Abstractor)
- Promise to include expanded related work section which cites and discusses relevant literature in GNN community.

[In fact, in the introduction (Eq 1), we draw analogy to a general message-passing framework to motivate our proposal exactly for this reason: first, we go general to discuss the motivation, then, we make a well-specified proposal. The contribution lies in the second step, whereas the first step serves a pedagagical purpose as motivation.]

---

[Used to be at the end of intro]

We summarize our responses below.
- The Zhang et al. paper is indeed relevant and will be appropriately cited and discussed in an expanded related work section. [However, we respectfully disagree with the characterization of our work that such work limits the novelty of ours, as will be explained below with respect to key and fundamental differences.]
- While GNNs and Transformers can be linked under the broader message-passing framework, they are distinct architectural paradigms that tackle different tasks and different domains. Our work is specifically about integrating explicit relational processing into the Transformer framework. [, which we believe ...]
- Although Zhang et al.'s proposed GNN architecture tackles a different domain to ours, it is still interesting to compare the computational mechanisms in each. We highlight some key differences in the response below.
- We remind the reviewer of the contributions of our work.
- We answer the reviewer's question on the use of learned symbolic attention. (yes)

## Relation to GNN Literature and message-passing paradigms, broadly

A defining feature of GNN models is the use of a form of *neural message-passing*. The neural message-passing framework is a highly general unifying framework that captures a very large class of architectures, including GNN architectures like GCNs, GAT, GraphSAGE, etc., and even Transformer models. This generality makes it a useful conceptual framework for thinking about different architectures in a common language, but it is not itself a well-specified architecture. The research challenge lies in the specific choices that make for an effective architecture for a particular class of tasks.
We point to the book by W. L. Hamilton, "Graph Representation Learning" (Chapter 5), for a nice exposition on the neural message-passing framework.

We recognize the usefulness of the message-passing framework as a conceptual tool, and in fact use it in the introduction section of our paper (Eq 1) to motivate our proposal by placing Transformers within the message-passing framework. ***But we emphasize that the originality of our work---and all work which can be formulated within this highly general framework---lies in the specific architectural design.***


## Relation to mentioned paper by Zhang et al., specifically

With the above discussion in mind, note that Zhang et al. (2024) begin with the generic message-passing framework in Eq 1 (as described elsewhere, e.g., in the Hamilton book), then in Eq2 proceed to write it differently in a generalized form to allow for propagation of edge features. The general form given in Eq2 in Zhang et al. is analogous to the general form in Eq1 in the introduction of our paper. But neither of these are concrete architectural proposals.

We note a few key differences in the architecture:
- The models operate over different data formats. Zhang et al.'s GNN model operates of graph inputs that consist of a set of node features, edges, *and edge features*. The edge features form the "starting edge features" at layer 1 in the same way that node features form the initial embedding. By contrast, our Transformer-based architecture is a *sequence* model, with no edges or edge features. This is a fundamental difference.
- The key proposal in Zhang et al. is to propose a mechanism for updating the edge features in the same way that typical message-passing operations update the node features. By contrast, we don't "update" the edge features, since they are not assumed to be an input to the model.
- In Zhang et al., the way the edge features are updated is simply as a linear map applied to the concatenation of the previous layer's edge features, and the pair of node features. We note that this is not a "relation" in the sense we use the term in our paper, because it does not include an explicit *comparison* between objects. Rather, it is a way to update the initial edge features (again, assumed given as input) as a function of the updated node features. In contrast, we compute relations at each layer as a series of inner product comparisons under different feature subspaces via different learned projection maps.