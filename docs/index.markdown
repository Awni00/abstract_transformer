---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<!-- <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script> -->

<!-- css for buttons -->
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<style>
.material-symbols-outlined {
  font-variation-settings:
  'FILL' 0,
  'wght' 400,
  'GRAD' 0,
  'opsz' 24
}
</style>
<!-- <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> -->
<style>
/* Style buttons */
.btn {
    background-color: DodgerBlue; /* Blue background */
    border: none; /* Remove borders */
    color: white; /* White text */
    padding: 12px 16px; /* Some padding */
    font-size: 16px; /* Set a font size */
    cursor: pointer; /* Mouse pointer on hover */
    border-radius: 5px; /* Add border radius */
    display: flex; /* Enable flex layout */
    align-items: center; /* Center vertically */
    justify-content: center; /* Center horizontally */
    ext-decoration: none;
}

/* Darker background on mouse-over */
.btn:hover {
    background-color: RoyalBlue;
    text-decoration: none;
    color: white;
}

.btn:visited {
    color: white;
}

/* Center buttons */
.button-container {
    display: flex;
    justify-content: center;
    gap: 10px;
}
</style>

<div style="text-align: center">
<h1> Disentangling and Integrating Relational and Sensory Information in Transformer Architectures </h1>

Awni Altabaa, John Lafferty <br>
Department of Statistics & Data Science <br>
Yale University
<!-- Awni Altabaa<sup>1</sup>, John Lafferty<sup>2</sup>
<br>
<sup>1</sup> Department of Statistics and Data Science, Yale University <br>
<sup>2</sup> Department of Statistics and Data Science, Wu Tsai Institute, Institute for Foundations of Data Science, Yale University -->
</div>

<br>


<div class="button-container">
    <a href="https://arxiv.org/abs/2405.16727" class="btn" target="_blank">
    <span class="material-symbols-outlined">description</span>&nbsp;Paper&nbsp;
    </a>
    <a href="https://github.com/Awni00/abstract_transformer/" class="btn" target="_blank">
    <span class="material-symbols-outlined">code</span>&nbsp;Code&nbsp;
    </a>
    <a href="#experiment-logs" class="btn">
    <span class="material-symbols-outlined">experiment</span>&nbsp;Experimental Logs&nbsp;
    </a>
</div>

<br>

<figure style="text-align: center;">
    <img src="figs/paper_main_figure.png" alt="A depiction of self-attention and relational attention">
</figure>

## Abstract

The Transformer architecture processes sequences by implementing a form of neural message-passing that consists of iterative information retrieval (attention), followed by local processing (position-wise MLP). Two types of information are essential under this general computational paradigm: "sensory" information about individual objects, and "relational" information describing the relationships between objects. Standard attention naturally encodes the former, but does not explicitly encode the latter. In this paper, we present an extension of Transformers where multi-head attention is augmented with two distinct types of attention heads, each routing information of a different type. The first type is the standard attention mechanism of Transformers, which captures object-level features, while the second type is a novel attention mechanism we propose to explicitly capture relational information. The two types of attention heads each possess different inductive biases, giving the resulting architecture greater efficiency and versatility. The promise of this approach is demonstrated empirically across a range of tasks.

## Summary of Paper

The Transformer architecture can be understood as an instantiation of a broader computational paradigm implementing a form of neural message-passing that iterates between two operations: 1) information retrieval (self-attention), and 2) local processing (feedforward block). To process a sequence of objects $$x_1, \ldots, x_n$$, this general neural message-passing paradigm has the form

$$
\begin{align*}
x_i &\gets \mathrm{Aggregate}(x_i, {\{m_{j \to i}\}}_{j=1}^n)\\
x_i &\gets \mathrm{Process}(x_i).
\end{align*}
$$

In the case of Transformers, the self-attention mechanism can be seen as sending messages from object $$j$$ to object $$i$$ that are encodings of the sender's features, with the message from sender $$j$$ to receiver $$i$$ given by $$m_{j \to i} = \phi_v(x_j)$$. These messages are then aggregated according to some selection criterion based on the receiver's features, typically given by the softmax attention scores.

We posit that there are essentially two types of information that are essential under this general computational paradigm: 1) *sensory* information describing the features and attributes of individual objects, and *relational* information about the relationships between objects. The standard attention mechanism of Transformers naturally encodes the former, but does not explicitly encode the latter.

In this paper, we propose *Relational Attention* as a novel attention mechanism which enables routing relational information between objects. We then introduce *Dual Attention*, a variant of multi-head attention combining two distinct attention mechanisms: 1) standard Self-Attention for routing sensory information, and 2) Relational Attention for routing relational information. This in turn defines an extension of the Transformer architecture with an explicit ability to reason over both types of information.

<figure style="text-align: center;">
    <img src="figs/dual_attn_alg.png" alt="An algorithmic description of Dual Attention.">
</figure>


## Experiments

We evaluate our proposed architecture on two sets of relational tasks: relational games and SET. We compare against previously proposed relational architectures, PrediNet and CoRelNet. We also compare against Transformers. Please see the paper for a description of the tasks and the experimental set up. We include a preview of the results here.


<!-- **Data-efficient Relational Reasoning: Relational Games.** ... -->
### Data-efficient Relational Reasoning: Relational Games

<div id="relational_games"></div>
<script>
fetch('figs/relgames_learning_curves.json')
    .then(response => response.json())
    .then(data => {
        var divElement = document.getElementById('relational_games');
        data.layout.width = divElement.offsetWidth;
        Plotly.react('relational_games', data.data, data.layout);
    });
</script>



<!-- ***Improved Symbolic Reasoning in Sequence-to-Sequence tasks: Mathematical Problem-Solving.*** -->
### Improved Symbolic Reasoning in Sequence-to-Sequence tasks: Mathematical Problem-Solving

<div id="mat"></div>
<script>
fetch('figs/math_training_curves.json')
    .then(response => response.json())
    .then(data => {
        var divElement = document.getElementById('math_');
        data.layout.width = divElement.offsetWidth;
        Plotly.react('math', data.data, data.layout).then(() => {
            MathJax.typesetPromise();
        });
    });

MathJax.Hub.Queue(["Typeset",MathJax.Hub,'contains_set_conv_rep']);
</script>

### Improvements in Language Modeling

<div id="language_modeling"></div>
<script>
fetch('figs/language_modeling_training_curves.json')
    .then(response => response.json())
    .then(data => {
        var divElement = document.getElementById('language_modeling');
        data.layout.width = divElement.offsetWidth;
        Plotly.react('language_modeling', data.data, data.layout);
    });
</script>

### The Benefits of Relational Inductive Biases in Vision: Image Recognition with ImageNet

<div id="vision"></div>
<script>
fetch('figs/imagenet_training_curves.json')
    .then(response => response.json())
    .then(data => {
        var divElement = document.getElementById('vision');
        data.layout.width = divElement.offsetWidth;
        Plotly.react('vision', data.data, data.layout);
    });
</script>


## Experiment Logs

Detailed experimental logs are publicly available. They include training and validation metrics tracked during training, test metrics after training, code/git state, resource utilization, etc.

**Relational games.** For code and instructions to reproduce the experiments, see [`this readme in the github repo`](https://github.com/Awni00/relational-convolutions/tree/main/experiments/relational_games). The experimental logs for each task can be found at the following links: [`same`](https://wandb.ai/awni00/relational_games-same), [`occurs`](https://wandb.ai/awni00/relational_games-occurs), [`xoccurs`](https://wandb.ai/awni00/relational_games-xoccurs), [`between`](https://wandb.ai/awni00/relational_games-1task_between), and [`match pattern`](https://wandb.ai/awni00/relational_games-1task_match_patt).

**SET.** For code and instructions to reproduce the experiments, see [`this readme in the github repo`](https://github.com/Awni00/relational-convolutions/tree/main/experiments/set). The experimental logs can be found [`here`](https://wandb.ai/awni00/relconvnet-contains_set).

## Citation

```
@article{altabaa2024disentangling,
    title={Disentangling and Integrating Relational and Sensory Information in Transformer Architectures},
    author={Awni Altabaa and John Lafferty},
    year={2024},
    journal={arXiv preprint arXiv:2402.08856}
}
```