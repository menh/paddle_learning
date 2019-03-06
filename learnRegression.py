# -*- utf-8
import paddle.fluid as fluid
import numpy as np
import paddle

x = fluid.layers.data(name = 'x', shape = [13], dtype = 'float32')
hidden = fluid.layers.fc(input = x, size = 100, act = 'relu')

net  = fluid.layers.fc(input = hidden, size = 1, act = None)

y = fluid.layers.data(name = 'y', shape = [1], dtype = 'float32')
cost = fluid.layers.square_error_cost(input = net, label =y)
avg_cost = fluid.layers.mean(cost)

test_program =fluid.default_main_program().clone(for_test = True)

optimizer = fluid.optimizer.SGDOptimizer(learning_rate = 0.01)
opts = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

x_data = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
test_data = np.array([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')


for pass_id in range(10):
    train_cost = exe.run(
        program = fluid.default_main_program(),
        feed = {'x': x_data, 'y': y_data},
        fetch_list = [avg_cost])
    print("Pass: %d, Cost: %0.5f" % (pass_id, train_cost[0]))

result = exe.run(program = test_program,
                feed = {'x': test_data, 'y': np.array([[0.0]]).astype('float32')},
                fetch_list = [net])
print (result)
print (" %0.5f:" % result[0][0][0])

import paddle.dataset.uci_housing as uci_housing

train_reader = paddle.batch(reader = uci_housing.train(), batch_size = 128)
test_reader = paddle.batch(reader = uci_housing.test(), batch_size = 128)

feeder = fluid.DataFeeder(place = place, fee_list = [x,y])
for pass_id in range(10):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program = fluid.default_main_program(),
                            feed = feeder.feed(data),
                            fetch_list = [avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))

    test_cost = 0
    for batch_id,data in enumerate(test_reader()):
        test_cost = exe.run(program = fluid.default_main_program(),
                            feed = feeder.feed(data),
                            fetch_list = [avg_cost])
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))
