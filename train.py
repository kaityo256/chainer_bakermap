#from future import print_function
import collections
import random

import chainer
from chainer import training
from chainer.training import extensions

import data
from model import Model

epoch = 100
batchsize = 100
gpu = -1


def main():
    units = 200
    ndata = 10000
    dataset = data.make_dataset(ndata, units)
    m = Model(units)
    model = m.get_model()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    test_ratio = 0.1
    nt = int(len(dataset)*test_ratio)
    test = dataset[:nt]
    train = dataset[nt:]
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    # Training
    trainer.run()
    m.save('baker.model')


if __name__ == '__main__':
    main()
