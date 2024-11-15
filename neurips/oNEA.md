
Thank you for your review. We refer you first to the global response for an overview of the additions and revisions made to address your concerns. Here, we provide more detail on your specific concerns.

**Comparison between *DAT* and the Abstractor (on math)**
> The experiments in this work are insufficient. For example, the experimental results in fig.3 are overall inferior to the Abstractor.

> In fig.3, except for algebra__sequence_next_term, the four types of mathematical problem-solving tasks perform poorly compared to Abstractors. It seems that the dual attention mechanism did not work well.

We are unsure why you think that the experimental results in Figure 3 are inferior to the Abstractor. The Abstractor paper also evaluates on the mathematics dataset considered in this work. Perhaps you are comparing Fig. 3 in our paper to Fig. 7 in the Abstractor paper? 

If so, the results of the two papers are not comparable since they are under different experimental settings. In particular, *the models in the Abstractor paper are much larger than the ones reported in Figure 3 in the current version of our paper*. The models in the Abstractor paper are roughly 2 million parameters in size (2.3M for Transformer and 2.2M for Abstractor). By contrast, in the current version of the paper, our models are roughly 800K parameters. (Note that the large model size of the Abstractor despite modest-seeming hyperparameters is due to the "multi-attention decoder" in the architecture that is used to attend alternatingly to the Encoder and Abstractor.)

We carried out additional experiments scaling up our models on the math experiments, some of which are shown in Table 1 in the uploaded 1-page pdf. While our current models are 2-layers deep, we carried out additional experimental runs with 3 layers (1M-parameter scale) and 4 layers (1.5M-parameter scale). Comparing the 1.5M-parameter scale *DAT* models to the results reported in the Abstractor paper (which has 2.2M parameters), *DAT* compares favorably.

In addition to allowing for comparison to the Abstractor results at comparable scales, these new experiments demonstrate the consistent superiority of *DAT* over a standard Transformer across models of varying sizes and depths. Moreover, we emphasize that the math experiments are just one set of experiments among a suite of experiments across different modalities and task paradigms. In particular, the *DAT* architecture is more general than the Abstractor (and, we argue, more natural, versatile, and scalable) and supports decoder-only language modeling---an important task paradigm that the Abstractor does not support. More on this below. We have also added the Abstractor as a baseline in the relational games experiments.

<!-- NOTE: if asks again about comparison, we can create figure/etc based on W&B results from the Abstractor paper. Don't need to recreate in pytorch. -->


**Additional baselines like PrediNet**
> In fig.2, why wasn’t PrediNet introduced for comparison? Since PrediNet is the baseline of relational games and has an attention mechanism.

We have conducted additional experimental runs for the relational games experiments evaluating learning curves on baselines including PrediNet, CoRelNet, and the Abstractor. We also added additional Transformer baselines increasing model size and number of attention heads to further support the claim that *DAT* outperforms Transformers on relational tasks.

We find that all relational architectures generally perform favorably compared to the Transformer baselines in terms of sample-efficiency. In our experiments, *DAT* outperforms PrediNet and the Abstractor in 4 out 5 relational games tasks. Expectedly, CoRelNet performs strongly on this benchmark. (CoRelNet has very strong inductive biases that encode the same/different relations that are a sufficient statistic for the relational games tasks, and was designed with benchmarks like this in mind.) Nonetheless, we observe that *DAT* compares favorably to relational architectures on synthetic relational tasks despite being a far more general and versatile modeling framework.

**The goal of our paper and how our suite of experiments supports the claims**

> The experiments in this work are insufficient.

We point first to the global response, where we explain the main goals and claims of the paper, and how the suite of experiments is designed to evaluate these claims.

In particular, we would like to highlight the new large-scale language modeling experiments that we added. Previously, our language modeling experiments consisted of modest-size models trained on the synthetically (LLM-)generated Tiny Stories dataset. We have greatly expanded our evaluation of *DAT* as a language model. In particular, we now evaluate models up to 1.3B-parameters in scale trained on the fineweb dataset (a real-world dataset of high-quality text scraped from the internet). Table 2 in the uploaded 1-page pdf reports the results from these experiments, showing that the relational computational mechanisms of *DAT* result in improvements in language modeling. This is an important finding and, we believe, a significant milestone in the line of work on relational architectures. So far, the success of relational architectures has been mostly limited to synthetic relational tasks (e.g., relational games). This result shows that these types of relational computational mechanisms can be integrated into a general and versatile modeling framework, granting enhanced relational processing without sacrificing general sequence modeling capabilities in other areas, and resulting in improvements even in complex real-world tasks like language understanding. We highlight Figure 2 in the 1-page pdf which presents a visualization of the relations learned in a *DAT* language model, which demonstrates interpretable semantic relations.

In addition to the language modeling experiments, our suite of experiments span multiple task paradigms and data modalities, evaluating the proposed architecture in a wide range of domains.

Responses to other, more minor concerns are in the official-comments.

---

# Response 1

> Thank you for your reply. I do not think that the performance of DAT in the relational game task is good, therefore, I will maintain my current rating.

We kindly request that you elaborate. Why do you think that the performance of DAT in the relational games task is not good? What would count as "good performance"? Please support your assertion with an explanation and references.

***Relational games experiments***

In the relational games experiments, expanded during the rebuttal, we compare *learning curves* between: (1) our *DAT* model, (2) multiple Transformer baselines of varying size and number of attention heads, (3) PrediNet, (4) CoRelNet, and (5) Abstractor. **We find that *DAT* consistently outperforms all Transformer baselines. Moreover, it also generally outperforms the PrediNet and Abstractor baselines, despite the fact that these are relational architectures with more rigid inductive biases whereas *DAT* is a general architectural framework that supports a wide range of task paradigms.** The significance of these experiments with respect to comparison to the relational architectures is that DAT performs competitively with these architectures despite the fact that these are narrow architectures with strong inductive biases designed specifically for synthetic relational tasks based on simple same/different relations.

**The main claim of our paper is that the *DAT* architecture possesses strong relational computational mechanisms without sacrificing the generality of the Transformer framework.** We emphasize that the aim of the relational games experiments---one of four distinct sets of experiments in our paper---is to demonstrate the effectiveness of *DAT*'s relational computational mechanisms in controlled synthetic settings, and to demonstrate competitiveness with narrow-domain relational architectures. **The results of the relational games experiments are fully consistent with this claim.**

---
***Following up on initial review and rebuttal***

Respectfully, both your initial review and this most-recent response lack an explanation for your criticisms, making it impossible for us to address any potential concerns you might have. We note that you did not respond to our rebuttal to your initial review. The full text of the "weaknesses" section of your initial review was as follows:
> The experiments in this work are insufficient. For example, the experimental results in fig.3 are overall inferior to the Abstractor. Since the computational mechanisms of DAT and Abstractor are similar, the authors should provide more reasons or explanations for such performance gaps.

In this statement, you asserted that our model is inferior to the Abstractor without providing an explanation. This was the only weakness outlined in the review. In our rebuttal, we explained why this is an **inaccurate assertion**. In particular, assuming your assertion was based on the results reported in the Abstractor paper (though you did not clarify this in your brief review), we explained that the results are not comparable due to the large difference model size ($2.2M$ vs $800K$ parameters; their model is almost 3x larger). We carried out extensive additional experiments on the mathematical problem-solving experiments performing a scaling analysisi and increasing the model size, in response to this comment in order to make a fair comparison. **When comparing models of comparable size (with *DAT* still being smaller), we find that *DAT* in fact outperforms the results reported in the Abstractor paper on several tasks in the mathematics benchmark.**

However, you did not respond to this from our rebuttal. We would appreciate your response or acknowledgment.

Now, you assert that "I do not think that the performance of DAT in the relational game task is good", again without providing an explanation. It is challenging for us to address your concerns if we do not receive engagement with our work.

---

***The full suite of experiments in our paper***

We note that the experimental evaluation in our paper includes a suite of four different sets of experiments spanning different paradigms and data modalities. In particular, our experiments include relational games, mathematical problem-solving, image recognition (ImageNet), and language modeling (Fineweb). Other reviewers have acknowledged this as being a "diverse", "strong", and "comprehensive" suite of experiments. In your review, you only make a surface-level reference to one of these experiments (and this reference is based on an inaccurate assertion, as explained above). We hope you will engage more deeply with our work and provide a full evaluation of all its contributions.

We point to the global response for an outline of the suite of experiments and how each set of experiments supports the main claims made in the paper.

---

We hope that you will respond and provide an explanation for your evaluation to facilitate a meaningful discussion and allow us to address any concerns. It's important that we all work together to ensure that the reviewing process in machine learning venues is as constructive and informative as possible.

---
