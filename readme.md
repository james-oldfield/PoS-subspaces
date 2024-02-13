# Parts of Speech–Grounded Subspaces in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2305.14053-red)](https://arxiv.org/abs/2305.14053) [![project_page](https://img.shields.io/badge/project_page-orange)](http://eecs.qmul.ac.uk/~jo001/PoS-subspaces/)

![main.jpg](./images/main.svg)
> CLIP represents multiple visual modes of variation in an embedding (e.g. the <span style="color:red">‘object’</span> and its <span style="color:blue">‘appearance’</span>). The learnt PoS subspaces more reliably separate the constituent visual components.

## Abstract

> **Parts of Speech–Grounded Subspaces in Vision-Language Models**<br>
James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis A. Nicolaou, and Ioannis Patras<br>
*ArXiv*, 2023 <br>
https://arxiv.org/abs/2305.14053 <br>
> **Abstract**: Latent image representations arising from vision-language models have proved immensely useful for a variety of downstream tasks. However, their utility is limited by their entanglement with respect to different visual attributes. For instance, recent work has shown that CLIP image representations are often biased toward specific visual properties (such as objects or actions) in an unpredictable manner. In this paper, we propose to separate representations of the different visual modalities in CLIP’s joint vision-language space by leveraging the association between parts of speech and specific visual modes of variation (e.g. nouns relate to objects, adjectives describe appearance). This is achieved by formulating an appropriate component analysis model that learns subspaces capturing variability corresponding to a specific part of speech, while jointly minimising variability to the rest. Such a subspace yields disentangled representations of the different visual properties of an image or text in closed form while respecting the underlying geometry of the manifold on which the representations lie. What’s more, we show the proposed model additionally facilitates learning subspaces corresponding to specific visual appearances (e.g. artists’ painting styles), which enables the selective removal of entire visual themes from CLIP-based text-to-image synthesis. We validate the model both qualitatively, by visualising the subspace projections with a text-to-image model and by preventing the imitation of artists’ styles, and quantitatively, through class invariance metrics and improvements to baseline zero-shot classification.

## Install

First, please install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

![main.jpg](./images/method.svg)
> Geometry-respecting subspaces isolate the visual variation associated with a specific part of speech.


One can compute the decomposition as follows (using the default 4 parts of speech), by first loading the vocabulary:

```python
from model import Model
M = Model(root_dir='../')

M.load_vocab(M.categories, root_dir=M.root_dir)
M.encode_vocab(M.categories)
```

And then learn the subspace of the ambient Euclidean space with:

```python
lam = 0.5
e = {'N': M.E['N'][:n].clone(), 'A': M.E['A'][:n].clone(), 'V': M.E['V'][:n].clone(), 'AV': M.E['AV'][:n].clone()}

# get the subspace, and the mean:
W, m = M.decompose(categories=['N', 'A', 'V', 'AV'], encodings=e, lam=lam)
```

Or in the tangent space with:

```python
Wt, mt = M.decompose_tangent(categories=['N', 'A', 'V', 'AV'], encodings=e, lam=lam)
```

## Experiments

See [`experiments/pos.ipynb`](experiments/pos.ipynb).

Add custom theme dictionaries in `experiments/custom_dict.py`

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{oldfield2023pos,
    title={Parts of Speech–Grounded Subspaces in Vision-Language Models},
    author={James Oldfield and Christos Tzelepis and Yannis Panagakis and Mihalis A. Nicolaou and Ioannis Patras},
    year={2023},
    eprint={2305.14053},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

**Please feel free to get in touch at**: `j.a.oldfield@qmul.ac.uk`
