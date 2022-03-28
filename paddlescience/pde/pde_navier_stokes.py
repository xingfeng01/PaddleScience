# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .pde_base import PDE

import sympy
import numpy as np


#class NavierStokes:
class NavierStokes(PDE):
    """
    Two dimentional time-independent Navier-Stokes equation  

    .. math::
        :nowrap:

        \\begin{eqnarray*}
            \\frac{\\partial u}{\\partial x} + \\frac{\\partial u}{\\partial y} & = & 0,   \\\\
            u \\frac{\\partial u}{\\partial x} +  v \\frac{\partial u}{\\partial y} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 u}{\\partial x^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 u}{\\partial y^2} + dp/dx & = & 0,\\\\
            u \\frac{\\partial v}{\\partial x} +  v \\frac{\partial v}{\\partial y} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 v}{\\partial x^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 v}{\\partial y^2} + dp/dy & = & 0.
        \\end{eqnarray*}

    Parameters
    ----------
        nu : float
            Kinematic viscosity
        rho : float
            Density

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.pde.NavierStokes(0.01, 1.0)
    """

    def __init__(self, nu=0.01, rho=1.0, dim=2, time_dependent=False):
        super(NavierStokes, self).__init__(
            dim + 1, time_dependent=time_dependent)

        if dim == 2 and time_dependent == False:

            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable
            u = sympy.Function('u')(x, y)
            v = sympy.Function('v')(x, y)
            p = sympy.Function('p')(x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(x)
            # momentum x equation
            momentum_x = u * u.diff(x) + u * v.diff(y)

            # variables in order
            self.independent_variable = [x, y]
            self.dependent_variable = [u, v, p]

            # order
            self.order = 2

            # equations
            self.equations = list()
            self.equations.append(continuty)
            # self.pdes.append(momentum_x)

    def discretize(self, time_nsteps=None):

        if self.time_dependent == False:
            return self
        else:
            return NavierStokesImplicit(self)


class NavierStokesImplicit(PDE):
    def __init__(self, ns):
        super(NavierStokesImplicit, self).__init__(dim + 1)

        self.time_dependent = True
        self.time_discretize_method = "implicit"
        self.nu = ns.nu
        self.rho = ns.rho
        self.bc = ns.bc

        if dim == 2:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable current time step: u^{n}, v^{n}, p^{n}
            u = sympy.Function('u')(x, y)
            v = sympy.Function('v')(x, y)
            p = sympy.Function('p')(x, y)

            # dependent variable previous time step: u^{n-1}, v^{n-1}, p^{n-1}
            u_1 = sympy.Function('u_1')(x, y)
            v_1 = sympy.Function('v_1')(x, y)
            p_1 = sympy.Function('p_1')(x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # dt
            self.dt = sympy.Symbol('dt')

            # continuty equation
            continuty = u.diff(x) + v.diff(x)
            continuty_rhs = 0

            # momentum
            momentum_x_rhs = 0

            # variables in order
            self.independent_variable = [x, y]
            self.dependent_variable = [u, v, p]
            self.dependent_variable_1 = [u_1, v_1, p_1]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.equations.append(continuty)

            self.rhs = list()
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)


if __name__ == "__main__":
    ns = NavierStokes()

    outs = np.zeros((10, 3))
    outs[:, 1] = 1.0
    outs[:, 2] = 2.0

    parser(ns, None, outs, None, None)
    #parser(ns.pdes[0])

    # def __init__(self, nu=0.01, rho=1.0, dim=2, time_dependent=False):
    #     super(NavierStokes, self).__init__(
    #         dim + 1, time_dependent=time_dependent)
    #     if dim == 2 and time_dependent == False:
    #         # continuty 
    #         self.add_item(0, 1.0, "du/dx")
    #         self.add_item(0, 1.0, "dv/dy")
    #         # momentum x
    #         self.add_item(1, 1.0, "u", "du/dx")
    #         self.add_item(1, 1.0, "v", "du/dy")
    #         self.add_item(1, -nu / rho, "d2u/dx2")
    #         self.add_item(1, -nu / rho, "d2u/dy2")
    #         self.add_item(1, 1.0 / rho, "dw/dx")
    #         # momentum y
    #         self.add_item(2, 1.0, "u", "dv/dx")
    #         self.add_item(2, 1.0, "v", "dv/dy")
    #         self.add_item(2, -nu / rho, "d2v/dx2")
    #         self.add_item(2, -nu / rho, "d2v/dy2")
    #         self.add_item(2, 1.0 / rho, "dw/dy")
    #     elif dim == 2 and time_dependent == True:
    #         # continuty 
    #         self.add_item(0, 1.0, "du/dx")
    #         self.add_item(0, 1.0, "dv/dy")
    #         # momentum x
    #         self.add_item(1, 1.0, "du/dt")
    #         self.add_item(1, 1.0, "u", "du/dx")
    #         self.add_item(1, 1.0, "v", "du/dy")
    #         self.add_item(1, -nu / rho, "d2u/dx2")
    #         self.add_item(1, -nu / rho, "d2u/dy2")
    #         self.add_item(1, 1.0 / rho, "dw/dx")
    #         # momentum y
    #         self.add_item(2, 1.0, "dv/dt")
    #         self.add_item(2, 1.0, "u", "dv/dx")
    #         self.add_item(2, 1.0, "v", "dv/dy")
    #         self.add_item(2, -nu / rho, "d2v/dx2")
    #         self.add_item(2, -nu / rho, "d2v/dy2")
    #         self.add_item(2, 1.0 / rho, "dw/dy")
    #     elif dim == 3 and time_dependent == False:
    #         # continuty 
    #         self.add_item(0, 1.0, "du/dx")
    #         self.add_item(0, 1.0, "dv/dy")
    #         self.add_item(0, 1.0, "dw/dz")
    #         # momentum x
    #         self.add_item(1, 1.0, "u", "du/dx")
    #         self.add_item(1, 1.0, "v", "du/dy")
    #         self.add_item(1, 1.0, "w", "du/dz")
    #         self.add_item(1, -nu / rho, "d2u/dx2")
    #         self.add_item(1, -nu / rho, "d2u/dy2")
    #         self.add_item(1, -nu / rho, "d2u/dz2")
    #         self.add_item(1, 1.0 / rho, "dp/dx")
    #         # momentum y
    #         self.add_item(2, 1.0, "u", "dv/dx")
    #         self.add_item(2, 1.0, "v", "dv/dy")
    #         self.add_item(2, 1.0, "w", "dv/dz")
    #         self.add_item(2, -nu / rho, "d2v/dx2")
    #         self.add_item(2, -nu / rho, "d2v/dy2")
    #         self.add_item(2, -nu / rho, "d2v/dz2")
    #         self.add_item(2, 1.0 / rho, "dp/dy")
    #         # momentum z
    #         self.add_item(3, 1.0, "u", "dw/dx")
    #         self.add_item(3, 1.0, "v", "dw/dy")
    #         self.add_item(3, 1.0, "w", "dw/dz")
    #         self.add_item(3, -nu / rho, "d2w/dx2")
    #         self.add_item(3, -nu / rho, "d2w/dy2")
    #         self.add_item(3, -nu / rho, "d2w/dz2")
    #         self.add_item(3, 1.0 / rho, "dp/dz")
    #     elif dim == 3 and time_dependent == True:
    #         # continuty 
    #         self.add_item(0, 1.0, "du/dx")
    #         self.add_item(0, 1.0, "dv/dy")
    #         self.add_item(0, 1.0, "dw/dz")
    #         # momentum x
    #         self.add_item(1, 1.0, "du/dt")
    #         self.add_item(1, 1.0, "u", "du/dx")
    #         self.add_item(1, 1.0, "v", "du/dy")
    #         self.add_item(1, 1.0, "w", "du/dz")
    #         self.add_item(1, -nu / rho, "d2u/dx2")
    #         self.add_item(1, -nu / rho, "d2u/dy2")
    #         self.add_item(1, -nu / rho, "d2u/dz2")
    #         self.add_item(1, 1.0 / rho, "dp/dx")
    #         # momentum y
    #         self.add_item(2, 1.0, "dv/dt")
    #         self.add_item(2, 1.0, "u", "dv/dx")
    #         self.add_item(2, 1.0, "v", "dv/dy")
    #         self.add_item(2, 1.0, "w", "dv/dz")
    #         self.add_item(2, -nu / rho, "d2v/dx2")
    #         self.add_item(2, -nu / rho, "d2v/dy2")
    #         self.add_item(2, -nu / rho, "d2v/dz2")
    #         self.add_item(2, 1.0 / rho, "dp/dy")
    #         # momentum z
    #         self.add_item(3, 1.0, "dw/dt")
    #         self.add_item(3, 1.0, "u", "dw/dx")
    #         self.add_item(3, 1.0, "v", "dw/dy")
    #         self.add_item(3, 1.0, "w", "dw/dz")
    #         self.add_item(3, -nu / rho, "d2w/dx2")
    #         self.add_item(3, -nu / rho, "d2w/dy2")
    #         self.add_item(3, -nu / rho, "d2w/dz2")
    #         self.add_item(3, 1.0 / rho, "dp/dz")
