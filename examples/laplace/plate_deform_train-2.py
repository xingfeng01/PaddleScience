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

cfg = psci.utils.parse_args()
paddle.enable_static()
psci.config.enable_prim()

if cfg is not None:
    # Geometry
    npoints = cfg['Geometry']['npoints']
    seed_num = cfg['Geometry']['seed']
    sampler_method = cfg['Geometry']['sampler_method']
    # Network
    epochs = cfg['Global']['epochs']
    num_layers = cfg['Model']['num_layers']
    hidden_size = cfg['Model']['hidden_size']
    activation = cfg['Model']['activation']
    # Optimizer
    learning_rate = cfg['Optimizer']['lr']['learning_rate']
    # Post-processing
    solution_filename = cfg['Post-processing']['solution_filename']
    vtk_filename = cfg['Post-processing']['vtk_filename']
    checkpoint_path = cfg['Post-processing']['checkpoint_path']
else:
    # Geometry
    npoints = 10201
    seed_num = 1
    sampler_method = 'uniform'
    # Network
    epochs = 20000
    num_layers = 5
    hidden_size = 20
    activation = 'tanh'
    # Optimizer
    learning_rate = 0.001
    # Post-processing
    solution_filename = 'output_plate'
    vtk_filename = 'output_plate'
    checkpoint_path = 'checkpoints'

paddle.seed(seed_num)
np.random.seed(seed_num)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-1,-1), extent=(1,1))
geo.add_boundary(name='left', criteria=lambda x, y: abs(x + 1) < 1e-5)
geo.add_boundary(name='right', criteria=lambda x, y: abs(x - 1) < 1e-5)
geo.add_boundary(name='top', criteria=lambda x, y: abs(y - 1) < 1e-5)
geo.add_boundary(name='down', criteria=lambda x, y: abs(y + 1) < 1e-5)

# discretize geometry
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)

# Euler Beam equation
pde = psci.pde.PlateEquilibrium(stiff=1, mass=1, rhs=0, time_dependent=False)

# set boundary condition
bc_l_w = psci.bc.Dirichlet('w', rhs=0)
bc_r_w = psci.bc.Dirichlet('w', rhs=0)
bc_t_w = psci.bc.Dirichlet('w', rhs=0)
bc_d_w = psci.bc.Dirichlet('w', rhs=0)

# add bounday and boundary condition
pde.add_bc("left", bc_l_w)
pde.add_bc("right", bc_r_w)
pde.add_bc("top", bc_t_w)
pde.add_bc("down", bc_d_w)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=2, num_outs=1, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=10)

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)
