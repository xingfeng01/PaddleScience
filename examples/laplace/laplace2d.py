# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddlescience as psci
import numpy as np
import paddle

import time

import jax, jax.numpy as jnp

# psci.config.set_compute_backend("jax")

paddle.seed(1)
np.random.seed(1)

# analytical solution 
ref_sol = lambda x, y: np.cos(x) * np.cosh(y)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
geo.add_boundary(
    name="around",
    criteria=lambda x, y: (y == 1.0) | (y == 0.0) | (x == 0.0) | (x == 1.0))

# discretize geometry
npoints = 10201
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# Laplace
pde = psci.pde.Laplace(dim=2)

# set bounday condition
bc_around = psci.bc.Dirichlet('u', rhs=ref_sol)

# add bounday and boundary condition
pde.add_bc("around", bc_around)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
nins = 2
nouts = 1
nlayers = 5
nhidden = 20

net = psci.network.FCNet(
    num_ins=nins,
    num_outs=nouts,
    num_layers=nlayers,
    hidden_size=nhidden,
    activation='tanh')

#################

w_array = []
b_array = []
for i in range(nlayers):
    if i == 0:
        shape = (nins, nhidden)
    elif i == (nlayers - 1):
        shape = (nhidden, nouts)
    else:
        shape = (nhidden, nhidden)
    w = np.random.normal(size=shape).astype('float32')
    b = np.random.normal(size=shape[-1]).astype('float32')
    w_array.append(w)
    b_array.append(b)

for i in range(nlayers):

    if psci.config._compute_backend == "jax":

        weight = []
        for i in range(nlayers):
            w = jnp.array(w_array[i], dtype="float32")
            b = jnp.array(b_array[i], dtype="float32")
            weight.append((w, b))
            if i < (nlayers - 1):
                weight.append(())
        net._weights = weight

    else:
        w_init = paddle.nn.initializer.Assign(w_array[i])
        b_init = paddle.nn.initializer.Assign(b_array[i])
        net.initialize(n=[i], weight_init=w_init, bias_init=b_init)

#################

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=5, checkpoint_freq=20)

exit()

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)

# MSE
# TODO: solution array to dict: interior, bc
cord = pde_disc.geometry.interior
ref = ref_sol(cord[:, 0], cord[:, 1])
mse2 = np.linalg.norm(solution[0][:, 0] - ref, ord=2)**2

n = 1
for cord in pde_disc.geometry.boundary.values():
    ref = ref_sol(cord[:, 0], cord[:, 1])
    mse2 += np.linalg.norm(solution[n][:, 0] - ref, ord=2)**2
    n += 1

mse = mse2 / npoints

print("MSE is: ", mse)
