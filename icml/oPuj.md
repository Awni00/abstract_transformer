# Review

Summary:
The authors introduce a modification/extension of the classical attention mechanism they term ‘dual attention’, which not only routes sensory information (as in classic SA) but adds a dedicated pathway to exchange relational information between tokens using its unique attention matrix – allowing for information flow which differs from the sensory one.

Claims And Evidence:
All major claims are in my opinion well justified.
The authors do a good job in articulating and introducing their angle on the problem step by step, starting from the well-known Transformer self-attention mechanism. The relational information is explicitly computed (i.e. inductive bias) and added to a symbolic identity, hence justifying the claim that relational information is exchanged. (Although, actual insights into what exactly is exchanged would enhance the paper – currently placed in limited form in the appendix.)

Methods And Evaluation Criteria:
The authors evaluate their method on four different tasks with corresponding evaluation criteria (acc for vision, perplexity for language, etc.);
The selection of the vision task could in my opinion be significantly improved – Image classification on CIFAR seems rather ill-suited to show the power of relational processing:

CIFAR is very object-centric and a single-object dataset, hence computation of relations between tokens might be rather straight-forward (around center of image) -- BUT more importantly:
Image classification: It can be enough to look at one or very few tokens of a CIFAR image and directly tell what the class would be; Hence, this eval seems ill-suited to me.
 There is a variety of other vision tasks where the benefit of relations between parts might be much more intuitive, e.g. multi-object detection, tracking, semantic segmentation, just to name a few.

Theoretical Claims:
I have checked the formulas and algorithms, and briefly read through the supporting evidence for Theorem 1 (appendix) – but couldn’t spot any obvious issues;

Experimental Designs Or Analyses:
As previously mentioned, I think the visual experiments could be significantly improved by choosing a task that requires relational modelling in a more obvious way that would be much more intuitive to the reader (e.g. multi-object detection, semantic segmentation, etc.); It is unclear to me if an instance-based task like classification would benefit from this, as one token might already be enough to determine the correct class label.

Also: Experimental analyses are often performed well but not necessarily contrasted to related methods – this has been deferred to the appendix, but might be better placed in the main paper for visibility and to provide the reader with appropriate context.

Supplementary Material:
The supplementary material in the form of the appendix nicely complements the paper and shows plenty of additional insights; especially Section C in terms of additional insights regarding experiments; Very important in terms of comparison to highly-relevant work is Section D!

Relation To Broader Scientific Literature:
Relation to relevant literature can and should be improved – A very important relationship to the work of Altabaa et al. [22] is discussed in the appendix in detail, but the main body of the paper severely lacks in terms of discussing and attributing the similarity; While indeed different, both method are (in terms of underlying idea, choice of Transformer architecture and modified attention mechanism) very closely related – which should be appropriately indicated already in Sections 2.2 and 2.3.

Essential References Not Discussed:
None that come to mind – but not extremely up-to-date in this particular area.

Other Strengths And Weaknesses:
Strengths:
Originality & Significance:

Clear motivation and step-by-step intro of the method based on a known shortcoming of a missing dedicated modelling mechanism of token-relationships within a Transformer architecture, addressing a known but important gap
Proposed method applicable to range of modalities due to the choice of a Transformer architecture and the preservation of its generality (in terms of attention)
Authors demonstrate the applicability via a range of experiments across different tasks/modalities
Clarity:

Explanations well-supported through a good mix of figures, algorithms and formulas
The paper is well written and easy to read and follow; several details moved to the appendix, but the paper provides a good level of depth to easily follow
Weaknesses:

Discussion of & comparison to related works is lacking in parts of the main text: The similarity to [22] is discussed in detail in the appendix, but should be indicated much earlier and in a clearer manner in Sections 2.2 and 2.3; Similarity, almost all experiments exclusively compare default Transformers with DAT – although for some, there are related works available that might even outperform (see appendix Figure 6); Could be discussed in the respective section to be ‘up-front’ with the reader (I don’t expect the method to outperform task-specialist-methods, but a comparison to these works (even if treated in a class of their own) would in my opinion help the reader to better place the proposed method’s strengths
Experiments on the visual task, i.e. classification on CIFAR, seems rather ill-chosen to support a claim of modelling relations; see previous comments & questions-section
Details on the ‘subspace’ comparison for relationship computation, i.e. l \in Rd could be extended -- see questions.
Other Comments Or Suggestions:
Typos:

L 177 right: to both computational mechanismS (plural)
L 354 left: it is useful TO consider.. (to missing)
Questions For Authors:
I’d like the authors to provide some more details and insights into the explicit relation computation between the feature maps: The authors state this operation is performed with “
”, and “for each 
, the feature maps ..” (l151 f) – producing a relation vector “across different features subspaces”.
 Are these subspaces particularly chosen? And if yes, how?
 How many comparisons are performed between two tokens?
As this is one of the main components of this approach (and differences to [22]), this aspect could be made a lot clearer to the reader.

Comments on the visual task: I’d like to know why the authors think that Image classification on CIFAR benefits from the relational modelling. It clearly does, as we can see in the results, but as I’ve mentioned previously: One token might already be enough to solve the entire task of classifying an image – so this choice doesn’t particularly feel well suited.
 Have the authors visualised what relations are modelled? If not, is this possible and could be included? Do they represent any ‘expected’/intuitive relations (e.g. parts of the object)?
 The visualisation provided for the language task in Figure 8 (appendix) is quite interesting; so sth similar for the image task would be a good addition;

Follow-up:
 Have the authors thought about evaluating their model on an alternative vision task that more intuitively require modelling of relations, e.g. multi-object detection or semantic segmentation? Why/why now?

TLDR; I think the paper presents an interesting approach, although some aspects could be improved (as detailled before); Depending on the response, I’m happy to consider further increasing my score.

Code Of Conduct: Affirmed.
Overall Recommendation: 3: Weak accept (i.e., leaning towards accept, but could also be rejected)

---
---

## Response

Thank you for your thoughtful and constructive review. We appreciate your positive feedback on the originality, significance, and clarity of our work. We are especially grateful for the time and effort you took to thoroughly engage with our work, reading the appendix and making note of typos. We'd also like to thank you for your specific and constructive feedback, which we believe has helped us improve the paper further. Below, we outline the key concerns you raised and attempt to address them.

---

**A1: Discussion & Experimental Comparison to Related Work on Relational Architectures (in main body of paper)**

> Experimental analyses are often performed well but not necessarily contrasted to related methods – this has been deferred to the appendix, but might be better placed in the main paper for visibility

> The similarity to [22] is discussed in detail in the appendix, but should be indicated much earlier

We agree that a more detailed discussion of related work in the main text would enhance clarity. Accordingly, we will use the additional page allowance to:
- Integrate the experimental comparison to previous relational architectures [20,21,22], currently in Appendix C, into the main text.
- Expand the discussion of Altabaa et al. [22], currently in Appendix D, to Sections 2.2 and 2.3.

---

**A2: Suitability of CIFAR for Evaluating Relational Processing in Vision**

> CIFAR is very object-centric and a single-object dataset

It is true that the CIFAR datasets contain a single object per image. We view the utility of the explicit relational processing mechanisms of our architecture as applying to processing and reasoning about visual relationships between object *parts* and different patches of the image. For example, this may allow the model to detect and represent different types of symmetries in the image or to represent visual similarities between object parts occurring in different locations in the image.

> It can be enough to look at one or very few tokens of a CIFAR image and directly tell what the class would be.

In our models, images are divided into small 4x4-pixel patches (64 tokens per image), making it unlikely that a single token would suffice for classification. Since individual tokens represent very small regions at early layers, the models do need to consider information from several tokens/patches, including the relationships between tokens. At those early layers, the relations visually compare different patches, which can perhaps be thought of as analogous to applying one patch as a "kernel" or "filter" to another patch in the image. At later layers, tokens may come to represent more global higher-level features, and the relations can represent higher-level relations between object parts. Indeed, the improved results we observe for the *DAT* architecture demonstrate the utility of the enhanced relational processing capabilities of our architecture, even for the simple CIFAR benchmark.

That said, we agree that more complex tasks (e.g., multi-object detection, tracking, semantic segmentation) would better showcase our architecture’s capabilities. We will add a discussion on this limitation and potential future directions.

> Have the authors visualised what relations are modelled? If not, is this possible and could be included? Do they represent any ‘expected’/intuitive relations (e.g. parts of the object)?

Following your suggestion, we visualized the relations learned by the *ViDAT* model. We find that some relations do appear to represent intuitive "visual similarity" relations between object parts.

To illustrate this, we provide an example of a visualization of the learned relations on an image of a truck at [Layer 0](https://postimg.cc/PNC0fdLX) and [Layer 4](https://postimg.cc/gXQSL6wY). The patch labeled "source" represents the reference token, and the value annotations indicate sigmoid-normalized relation activations $r_{ij}[\ell]$. The relation activations appear to be high for object parts that are visually similar, especially at earlier layers.

We will add this to the paper.

---

**A3: Questions**

> Are these subspaces particularly chosen? And if yes, how?

The 'feature subspaces' are learned via $W_q^{rel}, W_k^{rel}$ during training, rather than being predefined. These are separately learned weights from $W_{q,k}^{attn}$, which defines the selection criterion of the attention operation.

> How many comparisons are performed between two tokens?

This is a hyperparameter of the model, denoted $d_r$. For example, in the 1.3B-parameter language model, $d_r = 128$.

We will revise the main text to make this aspect clearer.
