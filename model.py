import collections
import struct

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_out)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class Model(object):
    def __init__(self, n_unit):
        self.unit = n_unit
        self.model = L.Classifier(MLP(n_unit, 2))

    def load(self, filename):
        chainer.serializers.load_npz(filename, self.model)

    def save(self, filename):
        chainer.serializers.save_npz(filename, self.model)

    def predictor(self, x):
        return self.model.predictor(x)

    def get_model(self):
        return self.model

    def export(self, filename):
        p = self.model.predictor
        l1W = p.l1.W.data
        l1b = p.l1.b.data
        l2W = p.l2.W.data
        l2b = p.l2.b.data
        d = bytearray()
        for v in l1W.reshape(l1W.size):
            d += struct.pack('f', v)
        for v in l1b:
            d += struct.pack('f', v)
        for v in l2W.reshape(l2W.size):
            d += struct.pack('f', v)
        for v in l2b:
            d += struct.pack('f', v)
        open(filename, 'w').write(d)
