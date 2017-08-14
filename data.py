import random
import numpy as np
import chainer

def make_baker(n):
    a = []
    x = random.random()
    for i in range(n):
        x = x * 3.0
        x = x - int(x) 
        a.append(x)
    return a

def make_random(n):
    a = []
    for i in range(n):
        a.append(random.random())
    return a

def make_data(ndata,units):
    data = []
    for i in range(ndata):
        a = make_baker(units)
        data.append([a,0])
    for i in range(ndata):
        a = make_random(units)
        data.append([a,1])
    return data

def make_dataset(ndata,units):
    data = make_data(ndata,units)
    random.shuffle(data)
    n = len(data)
    xn = len(data[0][0])
    x = np.empty((n,xn),dtype=np.float32)
    y = np.empty(n,dtype=np.int32)
    for i in range(n):
        x[i] = np.asarray(data[i][0])
        y[i] = data[i][1]
    return chainer.datasets.TupleDataset(x,y)

def main():
    dataset = make_dataset(2,3)
    print(dataset)

if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    main()
