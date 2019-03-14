import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt

def multilayer_perceptron(input):
    hidden1 = fluid.layers.fc(input = input, size = 100, act = 'relu')

    hidden2 = fluid.layers.fc(input = hidden1, size = 100, act = 'relu')

    fc = fluid.layers.fc(input = hidden2, size = 10, act = 'softmax')

    return fc

def convolutional_netural_network(input):
    conv1 = fluid.layers.conv2d(input = input,
                                num_filters = 32,
                                filter_size = 3,
                                stride = 1)

    pool1 = fluid.layers.pool2d(input = conv1,
                                pool_size = 2,
                                pool_stride = 1,
                                pool_type = 'max')

    conv2 = fluid.layers.conv2d(input = pool1,
                                num_filters = 64,
                                filter_size = 3,
                                stride = 1)
    pool2 = fluid.layers.pool2d(input = conv2,
                                pool_size = 2,
                                pool_stride = 1,
                                pool_type = 'max')
    fc = fluid.layers.fc(input = pool2, size = 10, act = 'softmax')
    return fc

image = fluid.layers.data(name = 'image', shape = [1,28,28], dtype = 'float32')
label = fluid.layers.data(name = 'label', shape = [1], dtype = 'int64')

# model = multilayer_perceptron(image)
model = convolutional_netural_network(image)

cost = fluid.layers.cross_entropy(input = model, label = label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input = model, label = label)

test_program = fluid.default_main_program().clone(for_test = True)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate = 0.001)
opts = optimizer.minimize(avg_cost)

train_reader = paddle.batch(mnist.train(), batch_size = 128)
test_reader = paddle.batch(mnist.test(), batch_size = 128)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(place = place, feed_list = [image, label])

for pass_id in range(5):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(
                                    program = fluid.default_main_program(),
                                    feed = feeder.feed(data),
                                    fetch_list = [avg_cost, acc])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                 (pass_id, batch_id, train_cost[0], train_acc[0]))
    test_accs = []
    test_costs = []

    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program = test_program,
                                      feed = feeder.feed(data),
                                      fetch_list = [avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])

    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
