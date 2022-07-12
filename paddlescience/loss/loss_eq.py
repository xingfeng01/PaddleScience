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

import paddle
import copy
from .loss_base import LossFormula, CompFormula


class EqLoss(LossFormula):
    def __init__(self, eq=None, output=None):
        super(EqLoss, self).__init__()
        self._loss = [self]

        if eq is not None:
            self._eq = eq

        if output is not None:
            self._net = output._net
            self._input = output._input

    def compute(self, pde, net, input, rhs=None):

        # compute outs, jacobian, hessian
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs_der(input, bs)

        # compute rst on left-hand side
        loss = 0.0
        for i in range(len(pde.equations)):
            formula = pde.equations[i]
            rst = cmploss.compute_formula(formula, input)

            # loss
            if rhs is None:
                loss += paddle.norm(rst, p=2) * self._loss_wgt[i]
            else:
                loss += paddle.norm(rst - rhs, p=2) * self._loss_wgt[i]

        return loss, cmploss.outs