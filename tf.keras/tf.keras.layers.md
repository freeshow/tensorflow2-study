## 1、tf.keras.layers.Flatten

### 1.1 函数定义

```python
tf.keras.layers.Flatten(
    data_format=None, **kwargs
)
```

If inputs are shaped `(batch,)` without a channel dimension, then flattening adds an extra channel dimension and output shapes are `(batch, 1)`.

### 1.2 Example

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
                        border_mode='same',
                        input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```