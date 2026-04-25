# Mechanistic intuition for why mixed-selectivity/polysemanticity supports flexible behaviour

[![Open Exercise Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m-j-wojcik/recurrence_intuition/blob/main/tutorial_exercises.ipynb)

A self-contained tutorial (~45 min) that builds intuition for **how recurrent neural networks represent nonlinear computations**, using the XOR problem and mixed selectivity as a running example.

### Mixed selectivity and polysemanticity

In both neuroscience and machine learning, individual neurons often respond to multiple variables rather than encoding a single clean feature. Neuroscientists call this **mixed selectivity** — neurons in prefrontal cortex fire for seemingly arbitrary conjunctions of stimuli, contexts, and actions. In the ML interpretability literature, the same phenomenon appears as **polysemanticity** — single neurons in large language models activate for unrelated concepts like DNA sequences, Korean text, and HTTP headers.

These are not bugs. This mixing creates high-dimensional population representations that can be flexibly read out by simple downstream circuits to support many different computations: it is what enables flexible, context-dependent behaviour. This tutorial builds intuition for why this is necessary, starting from the simplest case where mixed selectivity is required: the XOR problem.

## What you'll learn

Starting from the classic XOR problem, the tutorial guides you through why some computations are hard for linear readouts, how mixed selectivity solves this, and how these ideas play out in a continuous-time recurrent neural network — even before training.

| Section | Topic | Key idea |
|:--------|:------|:---------|
| **1** | XOR & mixed selectivity | Pure selectivity neurons can't solve XOR; one mixed-selectivity neuron breaks the linear separability barrier |
| **2** | Temporal decision task | The same XOR rule embedded in a richer setting: sequential stimuli, a distractor dimension, and a delayed decision |
| **3** | Building a CTRNN | The continuous-time RNN equation, discretisation, and an exercise to implement the recurrence step |
| **4** | Probing the untrained network | Linear SVM decoding of Colour, Shape, Width, and XOR from the hidden states across time |
| **5** | Why does the untrained network already represent XOR? | Structural vs. functional definitions of selectivity — masking experiments that reveal why connectivity ≠ selectivity in recurrent networks |

## Prerequisites

- Python (comfortable with numpy, basic PyTorch)
- Some familiarity with neural networks (what a weight matrix does, what ReLU is)
- No prior knowledge of computational neuroscience required

## Getting started

**Option 1 — Google Colab (recommended)**

Click the badge above. The notebook installs its own dependencies and runs standalone — no local setup needed.

**Option 2 — Local**

```bash
git clone https://github.com/m-j-wojcik/recurrence_intuition.git
cd recurrence_intuition
pip install numpy matplotlib seaborn torch scikit-learn plotly pydantic pyyaml
jupyter notebook tutorial_exercises.ipynb
```

## Repository structure

```
recurrence_intuition/
├── tutorial_exercises.ipynb   ← Start here
└── README.md
```

## Key references

- Rigotti, M., Barak, O., Warden, M. R., Wang, X. J., Daw, N. D., Miller, E. K., & Fusi, S. (2013). The importance of mixed selectivity in complex cognitive tasks. *Nature*, 497(7451), 585–590. [doi:10.1038/nature12160](https://doi.org/10.1038/nature12160)

- Fusi, S., Miller, E. K., & Rigotti, M. (2016). Why neurons mix: high dimensionality for higher cognition. *Current Opinion in Neurobiology*, 37, 66–74. [doi:10.1016/j.conb.2016.01.010](https://doi.org/10.1016/j.conb.2016.01.010)

- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). Toy Models of Superposition. *Transformer Circuits Thread*. [link](https://transformer-circuits.pub/2022/toy_model/index.html) 

## License

MIT
