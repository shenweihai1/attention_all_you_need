
## Reproduce Transformer
Goal: Reproduce Attention is all your need paper. 
  - Just evaluate Base model (big model is not necessary).
  - You should use PyTorch to implement this framework, but you can not use `torch.nn.Transformer` and `torch.nn.MultiheadAttention`. You should implement a `Transformer` using PyTorch's other existing functions.
  - During iterating code, you don't need to validate all your code, because I don't have a target server. The running server does not have GPU.

Deliverables:
  - Add a README.md to include all necessary important information about this project.
  - Include instructions and installations etc to show how to train the model on target server RTX 5090 (server version is `pytorch:1.0.2-cu1281-torch280-ubuntu2404`).
  - Make code clear and readable.