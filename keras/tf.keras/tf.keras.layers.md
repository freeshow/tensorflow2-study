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

## 2、tf.keras.layers.concatenate

### 2.1 函数定义

```python
tf.keras.layers.concatenate(
    inputs, axis=-1, **kwargs
)
```

Arguments:

- **`inputs`**: A list of input tensors (at least 2).
- **`axis`**: Concatenation axis.
- **`\**kwargs`**: Standard layer keyword arguments.

Returns:

A tensor, the concatenation of the inputs alongside axis `axis`.

### 2.2 Example

```python
import tensorflow.keras as keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
conatenated = keras.layers.concatenate([x1, x2])

out = keras.layers.Dense(4)(conatenated)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
model.summary()
```

`输出`:

```
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            [(None, 16)]         0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            [(None, 32)]         0                                            
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 8)            136         input_5[0][0]                    
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 8)            264         input_6[0][0]                    
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 16)           0           dense_5[0][0]                    
                                                                 dense_6[0][0]                    
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 4)            68          concatenate[0][0]                
==================================================================================================
Total params: 468
Trainable params: 468
Non-trainable params: 0
__________________________________________________________________________________________________
```

> 可以看到conatenate对最后一维进行了串联，通道数变成了8+8=16，可以指指定axis=x来指定空间的第x维串联。

## 3、tf.keras.layers.add

### 3.1 函数定义

```python
tf.keras.layers.add(
    inputs, **kwargs
)
```

Arguments:

- **`inputs`**: A list of input tensors (at least 2).
- **`\**kwargs`**: Standard layer keyword arguments.

Returns:

A tensor, the sum of the inputs.

### 3.2 Example

```python
import tensorflow.keras as keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
model.summary()
```

`输出`:

```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 16)]         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 32)]         0                                            
__________________________________________________________________________________________________
dense (Dense)                   (None, 8)            136         input_1[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8)            264         input_2[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (None, 8)            0           dense[0][0]                      
                                                                 dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 4)            36          add[0][0]                        
==================================================================================================
Total params: 436
Trainable params: 436
Non-trainable params: 0
```

> add层将dense_1层的输入和dense_2层的输入加在了一起，是张量元素内容相加。