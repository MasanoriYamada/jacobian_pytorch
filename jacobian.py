import torch
from torch import autograd
import numpy as np


def jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs.
    it strongly assume that first axis of inputs and outputs is batch direction

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size()[1:] + inputs.size()).reshape(-1, *inputs.size())
    outputssum = torch.sum(outputs, dim=0)
    for i, out in enumerate(outputssum.view(-1)):
        col_i = autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()
    jac = jac.reshape(outputs.size()[1:] + inputs.size())
    aa = np.array(range(jac.dim()))
    cut_point = len(outputs.size())
    new_axis = np.concatenate([np.roll(aa[:cut_point], 1), aa[cut_point:]])  # batchを先頭に
    return jac.permute(tuple(new_axis))
