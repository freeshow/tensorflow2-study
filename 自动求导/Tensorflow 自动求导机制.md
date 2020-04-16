## 自动求导机制 

在机器学习中，我们经常需要计算函数的导数。TensorFlow 提供了强大的 **自动求导机制** 来计算导数。在即时执行模式下，TensorFlow 引入了 `tf.GradientTape()` 这个 “求导记录器” 来实现自动求导。以下代码展示了如何使用 `tf.GradientTape()` 计算函数  $y(x) = x^2$  在 $x = 3$ 时的导数：

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y, y_grad])
```

输出:

```python
[array([9.], dtype=float32), array([6.], dtype=float32)]
```

这里 `x` 是一个初始化为 3 的 **变量** （Variable），使用 `tf.Variable()` 声明。与普通张量一样，变量同样具有形状、类型和值三种属性。使用变量需要有一个初始化过程，可以通过在 `tf.Variable()` 中指定 `initial_value` 参数来指定初始值。这里将变量 `x` 初始化为 `3.` 。变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数。

`tf.GradientTape()` 是一个自动求导的记录器。只要进入了 `with tf.GradientTape() as tape` 的上下文环境，则在该环境中计算步骤都会被自动记录。比如在上面的示例中，计算步骤 `y = tf.square(x)` 即被自动记录。离开上下文环境后，记录将停止，但记录器 `tape` 依然可用，因此可以通过 `y_grad = tape.gradient(y, x)` 求张量 `y` 对变量 `x` 的导数。

在机器学习中，更加常见的是对多元函数求偏导数，以及对向量或矩阵的求导。这些对于 TensorFlow 也不在话下。以下代码展示了如何使用 `tf.GradientTape()` 计算函数 $L(w, b)= \|Xw + b - y\|^2$ 在  $w = (1, 2)^T, b = 1$ 时分别对   $w, b$ 的偏导数。其中 $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}$

```python
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])
```

输出:

```python
[62.5, array([[35.], [50.]], dtype=float32), array([15.], dtype=float32)]
```

这里， `tf.square()` 操作代表对输入张量的每一个元素求平方，不改变张量形状。 `tf.reduce_sum()` 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量（可以通过 `axis` 参数来指定求和的维度，不指定则默认对所有元素求和）。TensorFlow 中有大量的张量操作 API，包括数学运算、张量形状操作（如 `tf.reshape()`）、切片和连接（如 `tf.concat()`）等多种类型。

从输出可见，TensorFlow 帮助我们计算出了


$$
\begin{align}
L((1, 2)^T, 1) &= 62.5  \\
\frac{\partial L(w, b)}{\partial w} |_{w = (1, 2)^T, b = 1} &= \begin{bmatrix} 35 \\ 50\end{bmatrix}  \\
\frac{\partial L(w, b)}{\partial b} |_{w = (1, 2)^T, b = 1} &= 15
\end{align}
$$


## 基础示例：线性回归 

基础知识和原理

- UFLDL 教程 [Linear Regression](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/) 一节。

考虑一个实际问题，某城市在 2013 年 - 2017 年的房价如下表所示：

| 年份 | 2013  | 2014  | 2015  | 2016  | 2017  |
| ---- | ----- | ----- | ----- | ----- | ----- |
| 房价 | 12000 | 14000 | 15000 | 16500 | 17500 |

现在，我们希望通过对该数据进行线性回归，即使用线性模型 $y = ax + b$ 来拟合上述数据，此处 `a` 和 `b` 是待求的参数。

首先，我们定义数据，进行基本的归一化操作。

```python
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
```

接下来，我们使用梯度下降方法来求线性模型中两个参数 `a` 和 `b` 的值。

回顾机器学习的基础知识，对于多元函数 $f(x)$ 求局部极小值，梯度下降的过程如下：

- 初始化自变量为  $x_0, k=0$ 

- 迭代进行下列步骤直到满足收敛条件：

  > - 求函数 $f(x)$ 关于自变量的梯度 $\nabla f(x_k)$ 
  > - 更新自变量：$x_{k+1} = x_{k} - \gamma \nabla f(x_k)$。这里  $\gamma$ 是学习率（也就是梯度下降一次迈出的 “步子” 大小）
  > - $k \leftarrow k+1$

接下来，我们考虑如何使用程序来实现梯度下降方法，求得线性回归的解 $\min_{a, b} L(a, b) = \sum_{i=1}^n(ax_i + b - y_i)^2$ NumPy 下的线性回归 

### Numpy 下的线性回归

机器学习模型的实现并不是 TensorFlow 的专利。事实上，对于简单的模型，即使使用常规的科学计算库或者工具也可以求解。在这里，我们使用 NumPy 这一通用的科学计算库来实现梯度下降方法。NumPy 提供了多维数组支持，可以表示向量、矩阵以及更高维的张量。同时，也提供了大量支持在多维数组上进行操作的函数（比如下面的 `np.dot()` 是求内积， `np.sum()` 是求和）。在这方面，NumPy 和 MATLAB 比较类似。在以下代码中，我们手工求损失函数关于参数 `a` 和 `b` 的偏导数，并使用梯度下降法反复迭代，最终获得 `a` 和 `b` 的值。

```python
a, b = 0, 0

num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)
```

然而，你或许已经可以注意到，使用常规的科学计算库实现机器学习模型有两个痛点：

- 经常需要手工求函数关于参数的偏导数。如果是简单的函数或许还好，但一旦函数的形式变得复杂（尤其是深度学习模型），手工求导的过程将变得非常痛苦，甚至不可行。
- 经常需要手工根据求导的结果更新参数。这里使用了最基础的梯度下降方法，因此参数的更新还较为容易。但如果使用更加复杂的参数更新方法（例如 Adam 或者 Adagrad），这个更新过程的编写同样会非常繁杂。

而 TensorFlow 等深度学习框架的出现很大程度上解决了这些痛点，为机器学习模型的实现带来了很大的便利。

### TensorFlow 下的线性回归 

TensorFlow 的 **即时执行模式** 与上述 NumPy 的运行方式十分类似，然而提供了更快速的运算（GPU 支持）、自动求导、优化器等一系列对深度学习非常重要的功能。以下展示了如何使用 TensorFlow 计算线性回归。可以注意到，程序的结构和前述 NumPy 的实现非常类似。这里，TensorFlow 帮助我们做了两件重要的工作：

- 使用 `tape.gradient(ys, xs)` 自动计算梯度；
- 使用 `optimizer.apply_gradients(grads_and_vars)` 自动更新模型参数。

```python
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)
```

在这里，我们使用了前文的方式计算了损失函数关于参数的偏导数。同时，使用`tf.keras.optimizers.SGD(learning_rate=1e-3)` 声明了一个梯度下降 **优化器** （Optimizer），其学习率为 1e-3。优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其 `apply_gradients()` 方法。

注意到这里，更新模型参数的方法 `optimizer.apply_gradients()` 需要提供参数 `grads_and_vars`，即待更新的变量（如上述代码中的 `variables` ）及损失函数关于这些变量的偏导数（如上述代码中的 `grads` ）。具体而言，这里需要传入一个 Python 列表（List），列表中的每个元素是一个 `（变量的偏导数，变量）` 对。比如上例中需要传入的参数是 `[(grad_a, a), (grad_b, b)]` 。我们通过 `grads = tape.gradient(loss, variables)` 求出 tape 中记录的 `loss` 关于 `variables = [a, b]` 中每个变量的偏导数，也就是 `grads = [grad_a, grad_b]`，再使用 Python 的 `zip()` 函数将 `grads = [grad_a, grad_b]` 和 `variables = [a, b]` 拼装在一起，就可以组合出所需的参数了。

### Python 的 `zip()` 函数

`zip()` 函数是 Python 的内置函数。用自然语言描述这个函数的功能很绕口，但如果举个例子就很容易理解了：如果 `a = [1, 3, 5]`， `b = [2, 4, 6]`，那么 `zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]` 。即 “将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表”，和我们日常生活中拉上拉链（zip）的操作有异曲同工之妙。在 Python 3 中， `zip()` 函数返回的是一个 zip 对象，本质上是一个生成器，需要调用 `list()` 来将生成器转换成列表。

[![](https://freeshow.oss-cn-beijing.aliyuncs.com/blog/zip.jpg)

在实际应用中，我们编写的模型往往比这里一行就能写完的线性模型 `y_pred = a * X + b` （模型参数为 `variables = [a, b]` ）要复杂得多。所以，我们往往会编写并实例化一个模型类 `model = Model()` ，然后使用 `y_pred = model(X)` 调用模型，使用 `model.variables` 获取模型参数。关于模型类的编写方式可见 [“TensorFlow 模型” 一章](https://tf.wiki/zh/basic/models.html)。

