# -*- utf-8

import paddle.fluid as fluid
import numpy as np
import paddle

x = fluid.layer.data(name = 'x',shape = [2], dtype = 'float32')

hidden  = fluid.layer.fc(input = x, size = 2, act = None)

output = fluid.layers.fc(input = hidden, size = 1, act = None)

y = fluid.layer.data(name = 'y', shape = [1], dtype = 'float32')

cost = fluid.layer.square_error_cost(intput = output, label = y)
avg_cost = fluid.layers.mean(cost)

test_program = fluid.default_main_program().clone(for_test = True)

optimizer = fluid.optimizer.SGDOptimizer(learning_rate = 0.01)
opts = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

x_data = np.array([[1.0,0.0],
                   [2.0,0.0],
                   [3.0,0.0]]).astype('float32')

y_data = np.array([2.0],[4.0],[6.0]).astype('float32')

test_data = np.array([[4.0,0.0]]).astype('float32')

for pass_id in range(10):
    train_cost
