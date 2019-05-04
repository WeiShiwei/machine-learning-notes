### Custom layers 残差连接块



```python
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])
```



### Automatic differentiation(自动微分) and gradient tape

​	tensorflow 提供了用于自动微分的API，来计算一个函数的导数。一种更接近数学的求导方法是：先写一个python函数，封装好对参数的运算。然后使用tf.contrib.eager.gradients_function 来创建一个函数计算上面封装好的函数的导函数（可指定对哪个参数求导）。同时，只要嵌套调用该函数，即可求高阶导。

```python
x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0
```



### Custom training: basics

#### Tensorflow中关于Tensor和Variable的理解

- Variable是可更改的（mutable），而Tensor是不可更改的。一个直接的例子就是Tensor不具有assign函数，而Variable含有。
- python和其他语言的API以及实现方式存在差异，本文只探讨general以及python方面的内容。
- Variable用于存储网络中的权重矩阵等变量，而Tensor更多的是中间结果等。
- Variable是会显示分配内存空间的（既可以是内存，也可以是显存），需要初始化操作（assign一个tensor），由Session管理，可以进行存储、读取、更改等操作。相反地，诸如Const, Zeros等操作创造的Tensor，是记录在Graph中，所以没有单独的内存空间；而其他未知的由其他Tensor操作得来的Tensor则是只会在程序运行中间出现。
- Tensor可以使用的地方，几乎都可以使用Variable。



#### Example: Fitting a linear model

This typically involves a few steps:

1. Define the model.
2. Define a loss function.
3. Obtain training data.
4. Run through the training data and use an "optimizer" to adjust the variables to fit the data.

#### Define the model

```python
class Model(object):
  def __init__(self):
    # Initialize variable to (5.0, 0.0)
    # In practice, these should be initialized to random values.
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0
```

#### Define a loss function

```python
def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))
```

#### Obtain training data

```python
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise
```

#### Define a training loop

```python
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)
```

```python
model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()
```

```
Epoch  0: W=5.00 b=0.00, loss=8.98478
Epoch  1: W=4.59 b=0.39, loss=6.08795
Epoch  2: W=4.26 b=0.70, loss=4.24368
Epoch  3: W=4.00 b=0.96, loss=3.06913
Epoch  4: W=3.80 b=1.16, loss=2.32085
Epoch  5: W=3.64 b=1.32, loss=1.84398
Epoch  6: W=3.51 b=1.46, loss=1.53998
Epoch  7: W=3.41 b=1.56, loss=1.34613
Epoch  8: W=3.33 b=1.65, loss=1.22248
Epoch  9: W=3.26 b=1.72, loss=1.14358
```