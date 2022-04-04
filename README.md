## PaLM - Pytorch

Implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a>

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
    dim = 512,
    num_tokens = 20000,
    dim_head = 64,
    depth = 12,
    heads = 8
)

tokens = torch.randint(0, 20000, (1, 2048))
logits = palm(tokens) # (1, 2048, 20000)
```

## Todo

- [ ] use Triton to add bias-less Layernorm + stable softmax

## Citations

```bibtex
@article{chowdhery2022PaLM,
  title   = {PaLM: Scaling Language Modeling with Pathways},
  author  = {Chowdhery, Aakanksha et al},
  year    = {2022}
}
```
