> tensorflow 交叉熵计算函数输入中的 logits 都不是 softmax 或 sigmoid 的输出，而是 softmax 或 sigmoid 函数的输入，因为它在函数内部进行 sigmoid 或 softmax 操作。
>
> 即 logits 就是神经网络模型中的 $W * X$ 矩阵

## 1. tf.nn.sigmoid_cross_entropy_with_logits

### 1.1 函数定义

```python
tf.nn.sigmoid_cross_entropy_with_logits(
    labels=None, logits=None, name=None
)
```

Args:

- **`labels`**: A `Tensor` of the same type and shape as `logits`.
  - `shape`:  [batch_size, num_classes], 单样本是[num_classes]， 和 logits 具有相同的 type(float)和shape的张量(tensor)
  - 即需要one-hot编码形式
- **`logits`**: A `Tensor` of type `float32` or `float64`.
- **`name`**: A name for the operation (optional).

Returns:

A `Tensor` of the same shape as `logits` with the componentwise logistic losses.

> output不是一个数，而是一个batch中每个样本的loss, 所以一般配合tf.reduce_mea(loss)使用

Raises:

- **`ValueError`**: If `logits` and `labels` do not have the same shape.

### 1.2 函数作用

Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive. For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.

即适用于每一个类别都是相不排斥的（代码中可以看出），例如，有的可以划到多个类别中，给你一张照片，同时包含大象和狗。

假定 `x = logits`, `z = labels`. The logistic loss is

```
  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + log(1 + exp(-x))
= x - x * z + log(1 + exp(-x))
```

当 $x<0$ 时，exp(-x) 趋于无穷，溢出，故，我们转换上面的公式：

```
  x - x * z + log(1 + exp(-x))
= log(exp(x)) - x * z + log(1 + exp(-x))
= - x * z + log(1 + exp(x))
```

因此，为了确保稳定性并避免溢出，实现使用此等效公式：

```
max(x, 0) - x * z + log(1 + exp(-abs(x)))
```

>在 Sigmoid/Softmax 函数的数值计算过程中，容易因输入值偏大发生数值溢出现象；在计算交叉熵时，也会出现数值溢出的问题。为了数值计算的稳定性， TensorFlow 中提供了一个统一的接口，将 Sigmoid/Softmax 与交叉熵损失函数同时实现，同时也处理了数值不稳定的异常，一般推荐使用这些接口函数，避免分开使用 Softmax 函数与交叉熵损失函数。  

### 1.3 示例

```python
import tensorflow as tf

# 5个样本三分类问题，且一个样本可以同时拥有多类
y = tf.constant([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,0]], dtype=tf.float32)
logits = tf.constant([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]], dtype=tf.float32)

y_pred = tf.math.sigmoid(logits)
E1 = -y*tf.math.log(y_pred) - (1-y)*tf.math.log(1-y_pred)
print(E1)

E2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
print(E2)
```

`输出：`

```
tf.Tensor(
[[6.1989022e-06 3.0485876e+00 2.1269276e+00]
 [3.0485876e+00 4.5419773e-05 1.3132617e+00]
 [1.3132617e+00 2.1269276e+00 6.7153242e-03]
 [1.8149957e-02 1.5023305e-03 1.4632827e+00]
 [3.0485876e+00 2.4756414e-03 1.3132617e+00]], shape=(5, 3), dtype=float32)
```

**输出的E1，E2结果相同**

## 2. tf.nn.weighted_cross_entropy_with_logits

```python
tf.nn.weighted_cross_entropy_with_logits(
    labels, logits, pos_weight, name=None
)
```

Args:

- `labels`: 和logits具有相同的type(float)和shape的张量(tensor)
- `logits`： 
  - 一个数据类型（type）是float32或float64;
  - shape: [batch_size,num_classes], 单样本是 [num_classes]
- `pos_weight`: 正样本的一个系数

This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`, allows one to trade off recall and precision by up- or down-weighting the cost of a positive error relative to a negative error.

A value `pos_weight > 1` decreases the false negative count, hence increasing the recall. Conversely setting `pos_weight < 1` decreases the false positive count and increases the precision. This can be seen from the fact that `pos_weight` is introduced as a multiplicative coefficient for the positive labels term in the loss expression:

```
labels * -log(sigmoid(logits)) * pos_weight +
    (1 - labels) * -log(1 - sigmoid(logits))
```

## 3. tf.nn.softmax_cross_entropy_with_logits

### 3.1 函数介绍

```python
tf.nn.softmax_cross_entropy_with_logits(
    labels, logits, axis=-1, name=None
)
```

Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class). For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.

它对于输入的logits先通过softmax函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得结果不至于溢出
它适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象。 
output不是一个数，而是一个batch中每个样本的loss,所以一般配合tf.reduce_mean(loss)使用。

> labels 为one-hot形式，不支持多目标，只支持一个label代表一个类

`logits` and `labels` must have the same dtype (either `float16`, `float32`, or `float64`).

If using exclusive `labels` (wherein one and only one class is true at a time),  see `sparse_softmax_cross_entropy_with_logits`.

### 3.2 示例

```python
import tensorflow as tf

# 5个样本三分类问题，且一个样本只可以有一个类
y = tf.constant([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]], dtype=tf.float32)
logits = tf.constant([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]], dtype=tf.float32)

E2 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
print(E2)
```

`输出：`

```
tf.Tensor([1.6878611e-04 1.0346780e-03 6.5883912e-02 2.6669841e+00 5.4985214e-02], shape=(5,), dtype=float32)
```

`注意与sigmoid_cross_entropy_with_logits 输出 shape 的区别`：

- sigmoid_cross_entropy_with_logits output shape=(5, 3) 
- softmax_cross_entropy_with_logits out  shape=(5,)

## 4. tf.nn.sparse_softmax_cross_entropy_with_logits

### 4.1 函数介绍

```python
tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels, logits, name=None
)
```

是 `tf.nn.softmax_cross_entropy_with_logits` 的简化版，labels 不需要进行One-hot编码

Args:

- `labels`：
  - 一个数据类型（type）是float32或float64;
  - shape:[batch_size, num_classes]
- `logits`: 一个数据类型（type）是float32或float64;

### 4.2 示例

```python
import tensorflow as tf

# 5个样本三分类问题，且一个样本只可以有一个类
y = tf.constant([ 0, 1, 2, 0, 1], dtype=tf.int32)
logits = tf.constant([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]], dtype=tf.float32)

E3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
print(E3)
```

`输出`:

```python
tf.Tensor([1.6878611e-04 1.0346780e-03 6.5883912e-02 2.5834920e+00 5.4985214e-02], shape=(5,), dtype=float32)
```

## 5. tf.keras.losses.categorical_crossentropy

### 5.1 定义

#### 5.1.1 `函数形式`：

```python
tf.keras.losses.categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0
)
```

Args:

- **`y_true`**: tensor of true targets.
- **`y_pred`**: tensor of predicted targets.
- **`from_logits`**: Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` encodes a probability distribution.
- **`label_smoothing`**: Float in [0, 1]. If > `0` then smooth the labels.

#### 5.1.2 `类形式`：

```python
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
    name='categorical_crossentropy'
)
```

```python
__call__(
    y_true, y_pred, sample_weight=None
)
```

###  5.2 示例

#### 5.2.1 函数形式

**Usage:**

```python
y_true = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=tf.float32)
y_pred = tf.constant([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]], dtype=tf.float32)

# 函数形式返回的是batch中每个example的 loss
loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
loss = tf.reduce_mean(loss)
print('Loss: ', loss.numpy())  # Loss: 0.0945
```

#### 5.2.2 类形式

**Usage:**

```python
y_true = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=tf.float32)
y_pred = tf.constant([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]], dtype=tf.float32)

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss = cce(y_true=y_true, y_pred=y_pred)
print('Loss: ', loss.numpy())  # Loss: 0.0945
```

> 可见，函数形式返回的是batch中每个example的 loss, 类形式返回的是整个batch的loss

**Usage with the `compile` API:**

```python
model = tf.keras.Model(inputs, outputs)
model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
```

## 6. tf.keras.losses.sparse_categorical_crossentropy

### 6.1 定义

#### 6.1.1 函数形式

```python
tf.keras.losses.sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, axis=-1
)
```

#### 6.1.2 类形式

```python
tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction=losses_utils.ReductionV2.AUTO,
    name='sparse_categorical_crossentropy'
)
```

> SparseCategoricalCrossentropy 即为 CategoricalCrossentropy的简化， labels不需要进行one-hot编码

### 6.2 示例

#### 6.2.1 函数形式

**Usage:**

```python
y_true = tf.constant([0, 1, 2], dtype=tf.int32)
y_pred = tf.constant([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]], dtype=tf.float32)
loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
loss = tf.reduce_mean(loss)
print('Loss: ', loss.numpy())  # Loss: 0.0945
```

### 6.2.2 类形式

**Usage:**

```python
y_true = tf.constant([0, 1, 2], dtype=tf.int32)
y_pred = tf.constant([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]], dtype=tf.float32)
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss = cce(y_true=y_true, y_pred=y_pred)
print('Loss: ', loss.numpy())  # Loss: 0.0945
```

**Usage with the `compile` API:**

```python
model = tf.keras.Model(inputs, outputs)
model.compile('sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

