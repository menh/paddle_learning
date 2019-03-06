import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np

def rnn_net(ipt, input_dim):
    emb = fluid.layers.embedding(input = ipt, size = [input_dim, 128], act = 'relu')
    sentence = fluid.layers.fc(input = emb, size = 128, act = 'tanh')

    tnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        prev = rnn.memory(shape = [128])
        hidden = fluid.layers.fc(input = [word, prev], size = 128, act = 'relu')
        run.update_memory(prev,hidden)
        run.output(hidden)

    last = fluid.layers.sequence_last_step(rnn())

    out = fluid.layers.fc(input = last, size = 2, act = 'softmax')
    return out
