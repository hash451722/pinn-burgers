import numpy as np
import torch


def func(x, y):
    f = 3*x**2 + 2*x*y + y**3
    return f

def func_pow(x,y):
    f = 3*torch.pow(x, 2) + 2*x*y + torch.pow(y, 3)
    return f


def f_grad():
    x = torch.tensor(4.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    f = func(x, y)
    # f = func_pow(x, y)


    f_x  = torch.autograd.grad(f, x, create_graph=True)[0]
    f_xx = torch.autograd.grad(f_x, x, create_graph=True)[0]

    f_y  = torch.autograd.grad(f, y, create_graph=True)[0]
    f_yy = torch.autograd.grad(f_y, y, create_graph=True)[0]

    f_xy = torch.autograd.grad(f_x, y, create_graph=True)[0]
    f_yx = torch.autograd.grad(f_y, x, create_graph=True)[0]

    print(f_x)
    print(f_xx)
    print(f_y)
    print(f_yy)
    print(f_xy)
    print(f_yx)


    z = torch.tensor(2.0, requires_grad=True)
    f_z  = torch.autograd.grad(f, z, create_graph=True, allow_unused=True)[0]
    print(f_z)


if __name__ == '__main__':
    f_grad()
