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

import paddle
from .algorithm_base import AlgorithmBase
from ..inputs import InputsAttr

from collections import OrderedDict
import numpy as np


class PINNs(AlgorithmBase):
    """
    The Physics Informed Neural Networks Algorithm.

    Parameters:
        net(NetworkBase): The NN network used in PINNs algorithm.
        loss(LossBase): The loss used in PINNs algorithm.

    Example:
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net, loss=loss)
    """

    def __init__(self, net, loss):
        super(PINNs, self).__init__()
        self.net = net
        self.loss = loss

    def create_inputs(self, pde):

        inputs = list()
        inputs_attr = OrderedDict()

        # TODO: hard code
        inputs_attr_i = OrderedDict()
        points = pde.geometry.interior
        data = points  # 
        inputs.append(data)
        inputs_attr_i["0"] = InputsAttr(0, 0)
        inputs_attr["interior"] = inputs_attr_i

        inputs_attr_b = OrderedDict()
        # loop on boundary
        for name in pde.bc.keys():
            data = pde.geometry.boundary[name]
            inputs.append(data)
            inputs_attr_b[name] = InputsAttr(0, 0)
        inputs_attr["boundary"] = inputs_attr_b

        return inputs, inputs_attr

    def create_labels(self, pde):

        labels = list()
        labels_attr = OrderedDict()

        # equation: rhs and weight
        #   - labels_attr["equation"][i]["rhs"]
        #   - labels_attr["equation"][i]["weight"]
        labels_attr["equations"] = list()
        for i in range(len(pde.equations)):
            attr = dict()
            rhs = pde.rhs_disc[i]
            weight = pde.weight_disc[i]

            if rhs is None:
                attr["rhs"] = None
            elif np.isscalar(rhs):
                attr["rhs"] = rhs
            elif type(rhs) is np.ndarray:
                labels.append(rhs)
                attr["rhs"] = rhs  # alias

            if weight is None:
                attr["weight"] = None
            elif np.isscalar(weight):
                attr["weight"] = weight
            elif type(weight) is np.ndarray:
                labels.append(weight)
                attr["weight"] = weight  # alias

            labels_attr["equations"].append(attr)

        # bc: rhs and weight
        #   - labels_attr["bc"][name_b][i]["rhs"]
        #   - labels_attr["bc"][name_b][i]["weight"]
        labels_attr["bc"] = OrderedDict()
        for name_b, bc in pde.bc.items():
            labels_attr["bc"][name_b] = list()
            for b in bc:
                attr = dict()
                rhs = b.rhs_disc
                weight = b.weight_disc
                if rhs is None:
                    attr["rhs"] = None
                elif np.isscalar(rhs):
                    attr["rhs"] = rhs
                elif type(rhs) is np.ndarray:
                    labels.append(rhs)
                    attr["rhs"] = rhs  # alias

                if weight is None:
                    attr["weight"] = None
                elif np.isscalar(weight):
                    attr["weight"] = weight
                elif type(weight) is np.ndarray:
                    labels.append(weight)
                    attr["weight"] = weight  # alias

            labels_attr["bc"][name_b].append(attr)

        return labels, labels_attr

    def compute(self, *inputs, inputs_attr, pde):

        # print(args[1])

        outs = list()

        # interior outputs and loss
        n = 0
        loss = 0.0
        for name_i, input_attr in inputs_attr["interior"].items():
            input = inputs[n]
            loss_i, out_i = self.loss.eq_loss(
                pde, self.net, name_i, input, input_attr,
                bs=-1)  # TODO: bs is not used
            loss += loss_i
            outs.append(out_i)
            n += 1

        # boundary outputs and loss
        for name_b, input_attr in inputs_attr["boundary"].items():
            input = inputs[n]
            loss_b, out_b = self.loss.bc_loss(
                pde, self.net, name_b, input, input_attr,
                bs=-1)  # TODO: bs is not used
            loss += loss_b
            outs.append(out_b)
            n += 1

        loss = paddle.sqrt(loss)
        return loss, outs  # TODO: return more
