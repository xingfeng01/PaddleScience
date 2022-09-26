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

import numpy as np
import matplotlib.pyplot as plt

import paddlescience as psci
import paddle

ref_rhs = lambda x: -np.sin(x) - 2 * np.sin(2 * x) - 3 * np.sin(3 * x) - 4 * np.sin(4 * x) - 8 * np.sin(8 * x)
ref_sol = lambda x: x + 1 / 8 * np.sin(8 * x) + 1 / np.sin(1 * x) + 1 / 2 * np.sin(2 * x) + 1 / 3 * np.sin(3 * x) + 1 / 4 * np.sin(4 * x)

geom = psci.geometry.Rectangular(origin=(0.), extent=(np.pi))
geom.add_boundary(name='left', criteria=lambda x: x == 0.)
geom.add_boundary(name='right', criteria=lambda x: x == np.pi)
geom_disc = geom.discretize(npoints=100, method='uniform')

pde = psci.pde.Poisson(dim=1, rhs=ref_rhs, weight=1.0)
bc_left = psci.bc.Dirichlet(name='u', rhs=ref_sol)
bc_right = psci.bc.Dirichlet(name='u', rhs=ref_sol)

pde.add_bc('left', bc_left)
pde.add_bc('right', bc_right)
pde = pde.discretize(geo_disc=geom_disc)

net = psci.network.FCNet(
    num_ins=1, num_outs=1, num_layers=3, hidden_size=20, activation='tanh')

loss = psci.loss.L2(p=2)
algo = psci.algorithm.PINNs(net=net, loss=loss)
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)
solution = solver.solve(num_epoch=1)
