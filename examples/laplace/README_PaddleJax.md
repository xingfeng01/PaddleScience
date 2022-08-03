
This readme shows how to algin precision with PaddlePaddle and Jax (in PaddleScience). There are two points

- Enable Jax as backend: add following line in beginning of code.
```
psci.config.set_compute_backend("jax")
```

- Align initial weights and biases with pre-created numpy.ndarray data. As shown in following example, firstly create w_array (numpy.ndarray) and b_array (numpy.ndarray). Then

    - PaddlePaddle: paddlescience.network.intialize()
    - Jax: create parameters.    
```

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

#################

w_array = []
b_array = []
for i in range(num_layers):
    if i == 0:
        shape = (2, hidden_size)
    elif i == (num_layers - 1):
        shape = (hidden_size, 1)
    else:
        shape = (hidden_size, hidden_size)
    w = np.random.normal(size=shape).astype('float32')
    b = np.random.normal(size=shape[-1]).astype('float32')
    w_array.append(w)
    b_array.append(b)

for i in range(num_layers):

    if psci.config._compute_backend == "jax":

        weight = []
        for i in range(num_layers):
            w = jnp.array(w_array[i], dtype="float32")
            b = jnp.array(b_array[i], dtype="float32")
            weight.append((w, b))
            if i < (num_layers - 1):
                weight.append(())
        net._weights = weight

    else:
        w_init = paddle.nn.initializer.Assign(w_array[i])
        b_init = paddle.nn.initializer.Assign(b_array[i])
        net.initialize(n=[i], weight_init=w_init, bias_init=b_init)

```
