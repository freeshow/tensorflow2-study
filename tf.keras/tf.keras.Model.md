## fit
```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
    use_multiprocessing=False, **kwargs
)
```

Arguments:

- `x`

  : Input data. It could be:

  - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
  - A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
  - A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
  - A [`tf.data`](https://tensorflow.google.cn/api_docs/python/tf/data) dataset. Should return a tuple of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.
  - A generator or [`keras.utils.Sequence`](https://tensorflow.google.cn/api_docs/python/tf/keras/utils/Sequence) returning `(inputs, targets)` or `(inputs, targets, sample weights)`. A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) is given below.

- **`batch_size`**: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 32. Do not specify the `batch_size` if your data is in the form of symbolic tensors, datasets, generators, or [`keras.utils.Sequence`](https://tensorflow.google.cn/api_docs/python/tf/keras/utils/Sequence) instances (since they generate batches).

> if your data is in the form of symbolic tensors, datasets, generators, or [`keras.utils.Sequence`](https://tensorflow.google.cn/api_docs/python/tf/keras/utils/Sequence) instances , 则不需要设置batch_size，因为这些数据集自动生成 batch。

