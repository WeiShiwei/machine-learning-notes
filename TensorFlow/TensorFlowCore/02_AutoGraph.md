------

[TOC]

# tf.function and AutoGraph in TensorFlow 2.0

https://www.tensorflow.org/beta/guide/autograph?hl=zh-cn

AutoGraph是TF提供的一个非常具有前景的工具, 它能够将一部分python语法的代码转译成高效的图表示代码. 由于从TF 2.0开始, TF将会默认使用动态图(eager execution), 因此利用AutoGraph, **在理想情况下**, 能让我们实现用动态图写(方便, 灵活), 用静态图跑(高效, 稳定).



## 1. tf.function示例

tf.function和AutoGraph通过生成代码并将其跟踪到TensorFlow图中来工作。 

### Download data

```python
def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

def mnist_dataset():
  (x, y), _ = tf.keras.datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds

train_dataset = mnist_dataset()
```



### Define the model

```python
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))
model.build()
optimizer = tf.keras.optimizers.Adam()
```



### Define the training loop

```python
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


@tf.function
def train(model, optimizer):
  train_ds = mnist_dataset()
  step = 0
  loss = 0.0
  accuracy = 0.0
  for x, y in train_ds:
    step += 1
    loss = train_one_step(model, optimizer, x, y)
    if step % 10 == 0:
      tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
  return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
```



## 2. 计算图

### 2.1 tf.function和AutoGraph的机制？

AutoGraph把python的代码转化成计算图。



### 2.2 为什么 TensorFlow 需要使用计算图呢？

计算图允许各种各样的优化，例如移除公共的子表达式和内核融合等。此外，计算图简化了分布式训练和部署时的环境配置，因此它们可被视为一种独立于平台的模型计算形式。这一特性对于在多 GPU 或 TPU 上的分布式训练极其重要，当然基于 TensorFlow Lite 在移动端和 IoT 上部署模型也非常重要。



## 3.  tf.function和Autograph使用指南

TODO

Functions can be faster than eager code, for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), you may not see much speedup.