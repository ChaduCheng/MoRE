import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import argparse

# parser = argparse.ArgumentParser(
#     description='deepfool',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
# parser.add_argument('--norm', '-n', default=None, type=str, choices=['2', 'inf'])
# parser.add_argument('--epsilon', '-e', default=None, type=float)
#
# args = parser.parse_args()


def l2_project(x, orig_input, eps):
    diff = x - orig_input
    diff = diff.renorm(p=2, dim=0, maxnorm=eps)
    return torch.clamp(orig_input + diff, 0, 1)


def linf_project(x, orig_input, eps):
    diff = x - orig_input
    diff = torch.clamp(diff, -eps, eps)
    return torch.clamp(diff + orig_input, 0, 1)


def project(new_input, orig_input, eps, norm):
    if norm == "2":
        return l2_project(new_input, orig_input, eps)
    elif norm == "inf":
        return linf_project(new_input, orig_input, eps)
    else:
        raise Exception("not implemented p norm.")


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50, norm="2", eps=1.0):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    f_image = net.forward(Variable(image[:, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[:], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            if norm == '2':
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
            elif norm == 'inf':
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(), ord=1)
            else:
                raise Exception("norm not specified!")
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        if norm == '2':
            r_i = abs(pert + 1e-4) * w / np.linalg.norm(w)
        elif norm == 'inf':
            r_i = abs(pert + 1e-4) * np.sign(w) / np.linalg.norm(w, ord=1)
        else:
            raise Exception("norm not specified!")
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        pert_image = project(pert_image, image, eps, norm)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image
