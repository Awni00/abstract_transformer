Score: 3

**Summary of criticism:**
- Missing discussion of parameter count in reported experiments. Issue of RA heads having more params than SA heads.
- Lack of baselines beyond Transformer

**Summary of responses:**
- We have added explicit parameter counts to all experiments and all figures, to make the relative model sizes clear. We also added additional baselines to the experiments such that there is a Transformer baseline that is *larger* than the *Dual Attention Transformer*.

> The main weakness of the paper is the poor quality of the empirical evaluation of the presented method. Basically, the experiments section is a mere ablation study comparing only variants of the proposed DAT, which includes the conventional transformer. Other baselines are completely missing.
- Criticism seems to be that we only include Transformer baselines in Language modeling. and that relational games are missing baselines like CoRelNet/PrediNet/ESBN/Abstractor/. and perhaps pure abstractor as baseline for mathematical problem-solving.


## The empirical claims of this paper and the suite of experiments presented

We would like to emphasize that the primary goal of this paper is to propose an extension of Transformers which supports relational inductive biases while retaining the generality of the Transformer framework, and being applicable in any context in which Transformers are applicable without a performance hit due to the stronger inductive biases (e.g., language modeling, Seq2Seq tasks, discriminative tasks, multiple modalities incl. vision, etc).

There exists a line of work studying relational inductive biases in neural architectures, producing architectures like PrediNet/CoRelNet/Abstractor/etc. One important finding of this line of work is that Transformers underperform in relational tasks in terms of sample efficiency and generalization. But these models are fairly narrow, and have only been applied to a small set of (typically discriminative) purely relational tasks. These models are much less general and versatile than Transformers and cannot be applied to many important settings in which Transformers excel. Among these prior works on relational inductive biases, the Abstractor is the most general, but still can only be applied to Seq2Seq tasks via a complex and cannot be directly applied to the important setting of language modeling. (We would be happy to elaborate on this in the discussion period.)

The main empirical claim of our work is that the proposed Dual Attention Transformer architecture realizes the benefits of relational inductive biases in relational tasks, compared to standard Transformers, and moreover that it retains the generality of the Transformer framework while enabling some performance benefits in complex real-world tasks (e.g., vision processing and language modeling) due to the enhanced relational processing capabilities. We argue that our suite of experiments demonstrate this claim as follows:
1. *Relational games.* this is a synthetic benchmark of relational tasks that has been used to evaluate previous relational architectures such as PrediNet and CoRelNet. Our results on this benchmark demonstrate that DAT does not exhibit the same limitations that the Transformer has been shown to exhibit in this line of work, and that DAT realizes the benefits of relational inductive biases on synthetic relational tasks in the same way as the narrow relational architectures like CoRelNet and PrediNet.
2. *Mathematical Problem-Solving.* This task demonstrates the capabilities of a sequence-to-sequence encoder-decoder architecture in  the dual attention transformer. The results of this experiment demonstrate that, like the Transformer, and encoder-decoder architecture is an effective method to model sequence-to-sequence tasks, thus demonstrating that DAT retains the generality of the Transformer in this aspect. This also serves as a point of comparison to the Abstractor, demonstrating that relational inductive biases result in improvements in mathematical/symbolic processing. [TODO: note that the Abstractor is not a baseline in this experiment, which is a point of criticism they might raise. be careful about this point]
3. *Language modeling on Tiny Stories (and now on large-scales with Fineweb with upto 1.3B-parameter models).* Language modeling is a crucial benchmark for general sequence modeling architectures which is outside the scope of previous narrow-domain architectures like PrediNet, CoRelNet, and even the Abstractor. 
4. *Image Recognition with ImageNet.*

> Basically, the experiments section is a mere ablation study comparing only variants of the proposed DAT, which includes the conventional transformer. Other baselines are completely missing.
It is true that a standard Transformer is technically a special case of our framework where the number of relational heads is zero. We disagree, however, that this makes this makes our experiments "merely an ablation study". Indeed, as explained above, the main goal of our paper is to propose a new class of architectures that supports relational inductive biases while being preserving the generality (with respect to task paradigms and modalities) of the Transformer framework. Thus, the experiments required to validate such claims are precisely the types of experiments we carried out in this paper. Comparison to RNN/SSM/etc baselines as you suggested for example, while indeed interesting, are tangential to the aim of this paper.

> Also the comparison of DAT variants seems somewhat unfair. The authors fix the number of total heads and then instantiate different shares of them with either conventional or relational attention. However, the relational attention head has about twice the parameters of a conventional attention head. Therefore, it is unclear whether the observed improvements are due to the proposed architectural change or just due to increased capacity
> Throughout the entire empirical evaluation I found no mention of the total number of parameters of the examined models. Since the presented method is supposed to have an inductive bias towards relational learning, the capacity trade-off is essential in the assessment of the method. Why is it missing?
> Not only is the investigation of the capacity trade-off missing, the relational attention heads have about twice the parameters of conventional attention. By fixing the number of heads rather than the number of parameters in their experiments, the authors obscure the fact that the comparison to the conventional Transformer is not at equal model capacity. How do you ensure that the observed improvements are due to the proposed architectural change and not simply due to increased model capacity?
- This criticism is about whether the parameter count of DAT is larger than T, and if so, is the comparison unfair? This is a fair criticism that we should address. We did think about this and incorporate this into our experiments (especially the more recent LM experiments), but we should discuss this explicitly.
- E.g., In Math experiments, we have a Transformer+ baseline that we didn't end up including for simplicity.
- We initially chose to emphasize the number of attention heads of each type in the comparison in our experiments to emphasize the effect of different inductive biases. For example, in the "purely relational" relational games experiments, to see whether the model can learn to use both relational and sensory heads, and to see whether the model could learn to choose between the different computational mechanisms available to it. Nonetheless, even though the parameter counts are similar in all baselines, we agree that it is important to explicitly mention this. Moreover, we agree that it is important to confirm that the difference in performance is not due to the small difference in model size. Thus, we have added descriptions/annotations of parameter count for all models in our experiments (both in the text and the figures), and have carried out additional experiments with Transformers that are *larger* than the corresponding *DAT* model.

TODOs:
- Add additional baselines to relational games experiments
- Add explicit mention of param count to all experiments. Maybe re-do some to do this more rigorously. e.g., by setting $n_h^{kv}$, as we did w language modeling.


---
Draft...

[Structure: summarize main criticisms of review; summarize our response in bulleted lists; address individual comments in more detail]

**Summary of review's main concerns and criticisms**

This review has two main points of criticism, which we agree are important, and hope to address:
1. Missing discussion of parameter count in reported experiments.
2. Missing baselines beyond Transformers.


**Summary of responses and relevant additions**
1. ...
2. ...

...


**Explanation of parameter count scaling**.
Explain that a SA head and RA head with the *same hyperparameter* are off by a factor of 1.5x (not 2x): i.e., $6 d^2: 4 d^2$. Moreover, when the relations are symmetric, the difference is smaller $5 d^2 : 4 d^2$ (i.e., 1.25x). Also, in a Transformer, there are more parameters in MLP than in MHA ($8 d^2$ for MLP vs $4 d^2$ for MHA). Moreover, in DAT, only a subset of the heads are relational heads. So, overall, the difference in parameter count is relatively small, even if the hyperparameters are chosen to be exactly the same.

We also control for parameter count / model capacity via the choice of hyper parameters (e.g., by increasing $d_{\mathrm{model}}$ for the Transformer model). We give an overview of this below, describing the parameter count in the models in each experiment. Overall, **the parameter count in the updated experiments are such that the Transformer baseline is the same size or even larger than the corresponding DAT model. We find the *DAT* continues to outperform the Transformer baseline, demonstrating that the performance benefit persists after explicitly controlling for model size.**