---
title: 最小二乘法的矩阵表示及非线性建模
date: 2018-06-26 13:43:19
tags: [机器学习]
category: [机器学习]
---

## 引言
早上上班的路上读《社会心理学》里面有一段话，觉得不错，摘在这里。

> **我们人类总是有一种不可抑制的冲动，想要解释行为，对其归因，以使其变得次序井然．具有可预见性，使一切尽在掌握之中。**你我对于类似的情境却可能表现出截然不同的反应，这是因为我们的想法不同。我们对朋友的责难做何反应，取决于我们对其所做的解释，取决于我们是把它归咎于朋友的敌意行为，还是归结于他糟糕的心情。
> 从某种角度来说，我们都是天生的科学家。我们解释着他人的行为，通常足够快也足够准确，以适应我们日常生活的需要。当他人的行为具有一致性而且与众不同时，我们会把其行为归因于他们的人格。例如。如果你发现一个人说话总是对人冷嘲热讽．你可能就会推断此人秉性不良，然后便设法尽量避免与他的接触。

当然这是指更大范围内的人类心理，在对数据和模型的痴迷上，人类的欲望显然也是强烈的，人类总是想尽一切的办法打破现有的桎梏，在创造了更多的不确定性之后，期望通过对数据的把控和预测以看到更确切的未来。

这次我们来从矩阵和向量的角度来解释最小二乘法及非线性的模拟建模。

## LS 的矩阵推导
继续使用上一次的方程式并将其改写成矩阵的形式如下：


$$
f(x_n;w_0,w_1) = w_0 + w_1x = w^Tx \tag{2.1}
$$

代入最小二乘损失函数，得到结果为：

$$
\begin{align}
\mathcal{L} &= \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}_n(t_n,f(x_n;w_0,w_1 )) \\
&= \frac{1}{N} \sum_{n=1}^{N} (t_n-w^tx_n)^2 \tag{2.2} \\
& = \frac{1}{N} \sum_{n=1}^{N} (t-Xw)^T(t-Xw) \tag{2.3}
\end{align}
$$

式 2.2 到式 2.3 的证明忽略。

其中 $X$ 、$W$ 和 $T$ 为：

$$
 X = \begin{bmatrix} x_1^T  \\ x_2^T  \\ x_3^T   \\ \vdots \\  x_n^T  \\ \end{bmatrix} =  \begin{bmatrix} 1 & x_1  \\ 1 &  x_2 \\ 1 & x_3  \\ \vdots & \vdots\\ 1 & x_n  \\ \end{bmatrix} \\
 w = \begin{bmatrix} w_0^T  \\ w_1^T \end{bmatrix} \quad \\
 t = \begin{bmatrix} t_1^T  \\ t_2^T  \\ t_3^T   \\ \vdots \\  t_n^T  \\ \end{bmatrix}
$$

对上述 2.3 展开，得到下面的表达式：

$$
\begin{align}
\mathcal{L} &= \frac{1}{N} w^TX^TXw - \frac {2} {N}w^TX^Tt + \frac {1}{N} t^Tt \\
& = \frac{1}{N} (w^TX^TXw - 2 w^TX^Tt + t^Tt)
\end{align}
$$

为了得到最小值，需要得到 $\mathcal{L}$的拐点（极小值）一致的向量$w$的值，这里一样是求 $\mathcal{L}$ 关于 $w$ 的偏导数，并令其为 0，可以代入上述的矩阵进行求解，也可以使用一些恒等式来直接化简:

$$
% inner array of minimum values 内层"最小值"数组
\begin{array}{c|c}
\hline
f(w) & 
\frac {\partial f} {\partial w} \\
\hline
w^Tx & x \\
x^t=Tw & x \\
w^Tw & 2w \\
w^TCw & 2Cw \\
\hline
\end{array}
$$

得到的表达式如下：

$$
\frac {\partial \mathcal{L}} {\partial w} = \frac {2} {N} X^TXw - \frac {2} {N}X^Tt = 0 
$$

$$
X^Xw = X^Tt \tag{2.4}
$$

从而得到使损失最小的$w$值，$\hat w$ 的矩阵公式为：

$$
\hat w = (X^TX)^{-1}X^Tt
$$

根据此公式解出的值与上次用标量形式解出的是一样的。




