---
training_framework: PaddlePaddle
---

# step1
```shell
cat ckpt.part.0* > ckpt.pt
```

# step2
```python3
import torch
torch.load("ckpt.pt")
```