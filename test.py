from model import Model
import numpy as np
import random
import data
import math

def main():
    ndata = 1000
    unit = 200
    model = Model(unit)
    model.load("baker.model")
    d = data.make_data(ndata,unit)
    x = np.array([v[0] for v in d], dtype=np.float32)
    y = model.predictor(x).data
    r = [np.argmax(v) for v in y]
    bs = sum(r[:ndata])
    rs = sum(r[ndata:])
    print("Check Baker")
    print "Success/Fail",ndata-bs,"/",bs
    print("Check Random")
    print "Success/Fail",rs,"/",ndata-rs

def test():
    unit = 200
    model = Model(unit)
    model.load("baker.model")
    a = []
    for i in range(unit):
        a.append(0.5)
    x = np.array([a], dtype=np.float32)
    y = model.predictor(x).data
    print(y)

if __name__ == '__main__':
    random.seed(2)
    np.random.seed(2)
    test()
    main()
