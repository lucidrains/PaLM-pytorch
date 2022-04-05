<img src="./palm.gif" width="450px"></img>

## PaLM - Pytorch

Implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a>, in less than 200 lines of code.

This model is pretty much SOTA on everything language.

It obviously will not scale, but it is just for educational purposes. To elucidate the public how simple it all really is.

## Install
```bash
$ pip install PaLM-pytorch
```

## Usage

```python
import torch
from palm_pytorch import PaLM

palm = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64,
)

tokens = torch.randint(0, 20000, (1, 2048))
logits = palm(tokens) # (1, 2048, 20000)
```

The PaLM 540B in the paper would be

```python
palm = PaLM(
    num_tokens = 256000,
    dim = 18432,
    depth = 118,
    heads = 48,
    dim_head = 256
)
```

## Test on Enwik8

```bash
$ python train.py
```

## Citations

```bibtex
@article{chowdhery2022PaLM,
    title   = {PaLM: Scaling Language Modeling with Pathways},
    author  = {Chowdhery, Aakanksha et al},
    year    = {2022}
}
```

```bibtex
@article{Tillet2019TritonAI,
    title   = {Triton: an intermediate language and compiler for tiled neural network computations},
    author  = {Philippe Tillet and H. T. Kung and David D. Cox},
    journal = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year    = {2019}
}
```
