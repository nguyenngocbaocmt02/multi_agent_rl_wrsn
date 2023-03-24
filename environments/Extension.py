import copy


def euclideanDistance(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** 0.5


def regularize(x):
    d = euclideanDistance(x, [0 for i in range(len(x))])
    if d == 0:
        return None
    for i in range(len(x)):
        x[i] /= d
    return x