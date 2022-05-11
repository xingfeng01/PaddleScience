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

paddle.seed(1)
np.random.seed(1)

# paddle.enable_static()
paddle.disable_static()

# load real data
real_data = np.load("flow_unsteady/flow_re200_10.00.npy").astype("float32")
real_cord = real_data[:, 0:3]
real_sol = real_data[:, 3:7]

cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -0.5),
    extent=(25, 8, 0.5),
    circle_center=cc,
    circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
geo_disc = geo.discretize(npoints=60000, method="sampling")

# N-S
pde = psci.pde.NavierStokes(
    nu=0.05,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[4.0, 0.01, 0.01, 0.01])
pde.set_time_interval([0.0, 10.0])

# boundary condition on left side: u=10, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=10.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# add initial condition
ic_u = psci.ic.IC('u', rhs=0.0)
ic_v = psci.ic.IC('v', rhs=0.0)
ic_w = psci.ic.IC('w', rhs=0.0)
pde.add_ic(ic_u, ic_v, ic_w)

# discretization
pde_disc = pde.discretize(time_step=0.5, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=4, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver train t0 -> t1
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

# print("###################### start time=0.5 train task ############")
uvw_t1 = solver.solve(num_epoch=2)
# uvw_t1 = uvw_t1[0]
# uvw_t1 = np.array(uvw_t1)
# print(uvw_t1.shape)

# # Solver train t1 -> tn
# time_step = 9
# current_uvw = uvw_t1[:, 0:3]
# for i in range(time_step):
#     current_time = 0.5 + (i + 1) * 0.5
#     print("###################### start time=%f train task ############" %
#           current_time)
#     solver.feed_data_n(current_uvw)  # add u(n)
#     real_xyzuvwp = GetRealPhyInfo(current_time)
#     real_uvwp = real_xyzuvwp[:, 3:7]
#     solver.feed_data(real_uvwp)  # add real data 
#     next_uvwp = solver.solve(num_epoch=2)
#     # Save vtk
#     psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=next_uvwp)
#     next_uvwp = next_uvwp[0]
#     next_uvwp = np.array(next_uvwp)
#     current_uvw = next_uvwp[:, 0:3]