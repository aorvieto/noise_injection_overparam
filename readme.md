
## Explicit Regularization in Overparametrized Models via Noise Injection
### Antonio Orvieto*, Anant Raj*, Hans Kersting* and Francis Bach

### Accepted at AISTATS 2023


Injecting noise within gradient descent has several desirable features, such as smoothing and regularizing properties. In this paper, we investigate the effects of injecting noise before computing a gradient step. We demonstrate that small perturbations can induce explicit regularization for simple models based on the $\ell_1$-norm, group $\ell_1$-norms, or nuclear norms. However, when applied to overparametrized neural networks with large widths, we show that the same perturbations can cause variance explosion. To overcome this, we propose using independent layer-wise perturbations, which provably allow for explicit regularization without variance explosion. Our empirical results show that these small perturbations lead to improved generalization performance compared to vanilla gradient descent.


Basically, just add these lines to your training loop befor computing the gradient:


```python
with torch.no_grad():
    i=0
    for param in model.parameters():
        param_copy.append(param.data)
        if i==(training_iteration%number_of_network_param_groups):
            param.data = param.data + sigma*torch.normal(0, 1, size=param.size(),device=device)
        i = i+1
```
