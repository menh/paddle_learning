# ֻ�����ۻ�������ʵ����������ֻ��д�㷨�����ÿ��
#����һ��ʱ�������ϵ�������������ѡ��һϵ�в��ͣ��������ѧϰ������Լ�����Ⳣ��

import paddle.fluid as fluid
import numpy as np

# ���峣��
x1 = fluid.layers.fill_constant(shape = [2, 2], value = 1, dtype = 'int64')
x2 = fluid.layers.fill_constant(shape = [2, 2], value = 1, dtype = 'int64')

# ��������
a = fluid.layers.create_tensor(dtype = 'int64', name = 'a')
b = fluid.layers.create_tensor(dtype = 'int64', name = 'b')

a1 = np.array([3, 2]).astype('int64')
b1 = np.array([1, 1]).astype('int64')

#��ֵ����
a2 = input('a = ')
b2 = input('b = ')

y1 = fluid.layers.sum(x = [a, b])



place = fluid.CPUPlace()
exe  =fluid.executor.Executor(place)
exe.run(fluid.default_startup_program())

out_a, out_b, result = exe.run(
                feed = {'a' : a2, 'b' : b2},
                fetch_list = [a, b, y1])
print (out_a, out_b, result)
