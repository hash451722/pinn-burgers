#  Physics-Informed Neural Network (PINN)
　微分方程式、初期条件、境界条件を損失関数として、微分方程式の解 u(t,x)を近似するニューラルネットネークを構築する手法である。

## PINNを用いたBurgers方程式の求解
　流体運動を記したNS方程式から圧力項を無視した一次元偏微分方程式であるBurgers' 方程式の解 u(t,x)を得る。Burgers' 方程式は次式で表される。

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x}
= \nu \frac{\partial^2 u}{\partial x^2}  
$$
$$
x \in (-1, 1)
$$
$$
t \in (0, 1]
$$

ここで、$u$は流速、$t$は時間、$x$は位置、$\nu$は動粘性係数である。


また、$t=0$の初期値を

$$
u(0,x) = -sin(\pi x)
$$

両端における境界値を

$$
u(t, -1) = u(t, 1) = 0
$$

とする。

また、Burgers' 方程式から$f(t, x)$を次式のように定義する。
$$
f(t, x) := \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2}
$$



## 損失関数

　初期値と境界値とニューラルネットの予測値$\hat{u}$との平均二乗誤差(MSE: Mean Squared Error)を$MSE_u$とする。

$$
MSE_u = \frac{1}{N_u}\sum_{i=1}^{N_u} | \hat{u}(t^i_u,x_u^i) - u^i|^2
$$


また、$f(t,x)$の平均二乗誤差を$MSE_f$として、

$$
MSE_f = \frac{1}{N_f}\sum_{i=1}^{N_f}|f(t_f^i,x_f^i)|^2
$$

これらの合計値を損失$MSE$とする。

$$
MSE = MSE_u + MSE_f
$$

$MSE$を最小化するようにニューラルネットの重みとバイアスを学習する。



## 実行環境



## Code

### 損失関数
https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html


### 最適化
https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html



---
## Link
https://blog.albert2005.co.jp/2020/07/10/numerical-simulation-through-deeplearning/

https://arxiv.org/abs/1711.10561


---

# YouTube
https://www.youtube.com/watch?v=zcgpIzaaKCo  
https://www.youtube.com/watch?v=B9ugHg9Sy6g
