# -*- utf-8

import paddle.fluid as fluid
import numpy as np
import paddle

x = fluid.layers.data(name = 'x',shape = [2], dtype = 'float32')

hidden  = fluid.layers.fc(input = x, size = 2, act = None)

output = fluid.layers.fc(input = hidden, size = 1, act = None)

y = fluid.layers.data(name = 'y', shape = [1], dtype = 'float32')

cost = fluid.layers.square_error_cost(input = output, label = y)
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

y_data = np.array([[2.0],[4.0],[6.0]]).astype('float32')

test_data = np.array([[4.0,0.0]]).astype('float32')

for pass_id in range(1000):
    train_cost = exe.run(
        program = fluid.default_main_program(),
        feed = {'x' : x_data, 'y' : y_data},
    #    fetch_list = [x_data])
    #print("Pass: %d, Cost: %0.5f" % (pass_id, train_cost[0]))

result = exe.run(program = test_program,
                 feed = {'x': test_data, 'y' : np.array([[0.0]]).astype('float32')},
                 fetch_list = [output])
print (result)
