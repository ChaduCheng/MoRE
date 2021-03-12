"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module contains the abstract class AttackerStep as well as a few subclasses. 

AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably do not need to
import or edit this module and can just think of it as internal.
"""

import torch as ch
import torch
import numpy as np
from utils_l1 import *
from torch.distributions import laplace
from torch.distributions import uniform
import random

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return ch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        l = len(x.shape) - 1
        rp = ch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return ch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """
    def project(self, x):
        """
        """
        return ch.clamp(x, 0, 1)

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=step_size)
        return ch.clamp(new_x, 0, 1)

class FourierStep(AttackerStep):
    """
    Step under the Fourier (decorrelated) parameterization of an image.

    See https://distill.pub/2017/feature-visualization/#preconditioning for more information.
    """
    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        return x

    def to_image(self, x):
        """
        """
        return ch.sigmoid(ch.irfft(x, 2, normalized=True, onesided=False))

class RandomStep(AttackerStep):
    """
    Step for Randomized Smoothing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_grad = False

    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + self.step_size * ch.randn_like(x)

    def random_perturb(self, x):
        """
        """
        return x

# L1 threat model
class L1Step(AttackerStep):
    """
    implementation is based on:
    https://advertorch.readthedocs.io/en/latest/advertorch/attacks.html#advertorch.attacks.L1PGDAttack
    https://github.com/BorealisAI/advertorch.git
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = batch_l1_proj(diff.cpu(), self.eps)
        if self.orig_input.is_cuda:
            diff = diff.cuda() 
        return ch.clamp(self.orig_input + diff, 0.0, 1.0)

    def step(self, x, g,l1_sparsity=0.95):
        """
        modified from perturb_iterative function, in iterative_projected_gradient.py
        from advertorch on github
        """
        # delta = torch.zeros_like(x, requires_grad = True)
        # return x  + self.step_size* l1_dir_topk(g.detach(),(x - self.orig_input), x,self.step_size,random.randint(5,20))
        grad = g
        abs_grad = ch.abs(grad)
        batch_size = g.size(0)
        view = abs_grad.view(batch_size, -1)
        view_size = view.size(1)
        
        if l1_sparsity is None:
                vals, idx = view.topk(1)
        else:
            vals, idx = view.topk(int(np.round((1 - l1_sparsity) * view_size)))
        out = ch.zeros_like(view).scatter_(1, idx, vals)
        out = out.view_as(grad)
        grad = grad.sign() * (out > 0).float()
        
        grad = normalize_by_pnorm(grad, p=1)


        return x + grad * self.step_size

    def random_perturb(self, x):
        """
        modified from perturb function, in iterative_projected_gradient.py
        from advertorch on github
        """
        delta = ch.zeros_like(x)
        delta = ch.nn.Parameter(delta)
        # delta = rand_init_delta(delta, x, 1, self.eps, 0, 1)
        delta = rand_init_delta(delta=delta, x=x, ord=1, eps=self.eps, clip_min=0.0, clip_max=1.0)
        # delta = ch.clamp(x + delta, min=self.clip_min, max=self.clip_max) - x
        return ch.clamp(x + delta, 0, 1)

############
def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 20) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval.half()).half() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)

def proj_l1ball(x, epsilon=10, device = "cuda:1"):
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
#         y = x* epsilon/norms_l1(x)
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y = y.view(-1,3,32,32)
    y *= x.sign()
    return y
def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).half().to(device)
    comp = (vec > (cssv - s)).half()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.half() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w