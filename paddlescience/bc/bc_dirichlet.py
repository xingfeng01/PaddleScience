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


class BC:
    def __init__(self, name):
        self.category = None
        self.name = name


class Free(BC):
    def __init__(self, name):
        super(Free, self).__init__(anme)
        self.category = "Free"

    def compute(self, u, du=None, dn=None, value=None):
        return 0.0


class Dirichlet(BC):
    def __init__(self, name):
        super(Dirichlet, self).__init__(name)
        self.category = "Dirichlet"

    def compute(self, u, du=None, dn=None, value=None):
        return paddle.norm(u - value, p=2)

    def equation(self, us, rh):
        return us


class Neumann(BC):
    def __init__(self, name):
        super(Neumann, self).__init__(name)
        self.category = "Neumann"

    # dn: normal direction
    def compute(self, u, du=None, dn=None, value=None):
        return paddle.norm(du * dn - value, p=2)

    def equation(self, us):
        return us.diff(n)


class Robin(BC):
    def __init__(self, name):
        super(Robin, self).__init__(name)
        self.category = "Robin"

    def compute(self, u, du=None, dn=None, value=None):
        diff = u + du * dn - value
        return paddle.norm(diff, p=2)

    def equation(self, us):
        return us + us.diff(n)


if __name__ == "__main__":

    # set geometry and boundary
    geo = pcsi.geometry.Rectangular(origine=(0.0, 0.0), extent=(1.0, 1.0))
    top = geo.add_boundary(name="top", lambda x, y: y == 1.0)
    dow = geo.add_boundary(name="down", lambda x, y: y == 0.0)

    # define N-S
    pde = psci.pde.NavierStokes(nu=0.1, rho=1.0, dim=2, time_dependent=False)

    # set bounday condition
    bctop_u = psci.bc.Dirichlet('u', 0)
    bctop_v = psci.bc.Dirichlet('v', 0)

    # bounday and bondary condition to pde
    pde.add_bc(top, bctop_u, bctop_v)
