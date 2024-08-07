
Thank you for your review. We refer you first to the global response for an overview of the additions and revisions made to address your concerns. Here, we provide more detail on your specific concerns.

**Comparison between *DAT* and the Abstractor (on math)**
> The experiments in this work are insufficient. For example, the experimental results in fig.3 are overall inferior to the Abstractor.

> In fig.3, except for algebra__sequence_next_term, the four types of mathematical problem-solving tasks perform poorly compared to Abstractors. It seems that the dual attention mechanism did not work well.

We are unsure why you think that the experimental results in Figure 3 are inferior to the Abstractor. The Abstractor paper also evaluates on the mathematics dataset considered in this work. Perhaps you are comparing Fig. 3 in our paper to Fig. 7 in the Abstractor paper? 

If so, the results of the two papers are not comparable since they are under different experimental settings. In particular, *the models in the Abstractor paper are much larger than the ones reported in Figure 3 in the current version of our paper*. The models in the Abstractor paper are roughly 2 million parameters in size (2.3M for Transformer and 2.2M for Abstractor). By contrast, in the current version of the paper, our models are roughly 800K parameters. (Note that the large model size of the Abstractor despite modest-seeming hyperparameters is due to the "multi-attention decoder" in the architecture that is used to attend alternatingly to the Encoder and Abstractor.)

We carried out additional experiments scaling up our models on the math experiments, some of which are shown in Table 1 in the uploaded 1-page pdf. While our current models are 2-layers deep, we carried out additional experimental runs with 3 layers (1M-parameter scale) and 4 layers (1.5M-parameter scale). Comparing the 1.5M-parameter scale *DAT* models to the results reported in the Abstractor paper (which has 2.2M parameters), *DAT* compares favorably.

In addition to allowing for comparison to the Abstractor results at comparable scales, these new experiments demonstrate the consistent superiority of *DAT* over a standard Transformer across models of varying sizes and depths. Moreover, we emphasize that the math experiments are just one set of experiments among a suite of experiments across different modalities and task paradigms. In particular, the *DAT* architecture is more general than the Abstractor (and, we argue, more natural, versatile, and scalable) and supports decoder-only language modeling---an important task paradigm that the Abstractor does not support. More on this below. We have also added the Abstractor as a baseline in the relational games experiments.


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

