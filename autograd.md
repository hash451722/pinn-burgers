# PyTorch autograd を使って微分係数を求める


[TORCH.AUTOGRAD.GRAD](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)を使って関数の微分係数を求める。


## 偏微分

$$
f(x,y) = 3x^2 + 2xy + y^3
$$
$$
\frac{\partial f(x,y)}{\partial x} = 6x + 2y
$$
$$
\frac{\partial^2 f(x,y)}{\partial x^2} = 6
$$
$$
\frac{\partial f(x,y)}{\partial y} = 2x + 3y^2
$$
$$
\frac{\partial^2 f(x,y)}{\partial y^2} = 6y
$$
$$
\frac{\partial^2 f(x,y)}{\partial x \partial y} = 2
$$
$$
\frac{\partial^2 f(x,y)}{\partial y \partial x} = 2
$$


```
import torch


def func(x, y):
    f = 3*x**2 + 2*x*y + y**3
    return f


def f_grad():
    x = torch.tensor(4.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    f = func(x, y)

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


if __name__ == '__main__':
    f_grad()
```
```
tensor(30., grad_fn=<AddBackward0>)
tensor(6., grad_fn=<MulBackward0>)
tensor(35., grad_fn=<AddBackward0>)
tensor(18., grad_fn=<MulBackward0>)
tensor(2.)
tensor(2.)
```

---

関数に含まれない変数で偏微分する場合、torch.autograd.gradに```allow_unused=True```を指定しておけばエラーにならずにNoneが返される。

```
z = torch.tensor(2.0, requires_grad=True)
f_z  = torch.autograd.grad(f, z, create_graph=True, allow_unused=True)[0]
print(f_z)
```
```
None
```
