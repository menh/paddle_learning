import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image


# ����������֪��
def multilayer_perceptron(input):
    # ��һ��ȫ���Ӳ㣬�����ΪReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # �ڶ���ȫ���Ӳ㣬�����ΪReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # ��softmaxΪ�������ȫ���������㣬��СΪlabel��С
    fc = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return fc


# ������������
def convolutional_neural_network(input):
    # ��һ�������㣬�����˴�СΪ3*3��һ����32��������
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    # ��һ���ػ��㣬�ػ���СΪ2*2������Ϊ1�������ػ�
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # �ڶ��������㣬�����˴�СΪ3*3��һ����64��������
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)

    # �ڶ����ػ��㣬�ػ���СΪ2*2������Ϊ1�������ػ�
    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # ��softmaxΪ�������ȫ���������㣬��СΪlabel��С
    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc


# ����������
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# ��ȡ������
# model = multilayer_perceptron(image)
model = convolutional_neural_network(image)

# ��ȡ��ʧ������׼ȷ�ʺ���
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# ��ȡѵ���Ͳ��Գ���
test_program = fluid.default_main_program().clone(for_test=True)

# �����Ż�����
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

# ��ȡMNIST����
train_reader = paddle.batch(mnist.train(), batch_size=128)
test_reader = paddle.batch(mnist.test(), batch_size=128)

# ����һ��ʹ��CPU��ִ����
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# ���в�����ʼ��
exe.run(fluid.default_startup_program())

# ������������ά��
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# ��ʼѵ���Ͳ���
for pass_id in range(10):
    # ����ѵ��
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # ÿ100��batch��ӡһ����Ϣ
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # ���в���
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # �����Խ�����ƽ��ֵ
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))


# ��ͼƬ����Ԥ����
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


# �������ݲ���ʼԤ��
img = load_image('image/infer_3.png')
results = exe.run(program=test_program,
                  feed={'image': img, "label": np.array([[1]]).astype("int64")},
                  fetch_list=[model])
# ��ȡ���������ı�ǩ
lab = np.argsort(results)[0][0][-1]
print('infer_3.png infer result: %d' % lab)
