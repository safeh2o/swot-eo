import numpy as np

def con(x,*idx):
    return x[idx[0]]-x[idx[1]]

def SSE(y_true,y_pred):
    SSE = np.sum(np.ones(len(y_true)) * (y_pred - y_true) ** 2)
    return SSE

'''def MAE(y_true,y_pred):
    MAE= np.nanmean(np.ones(len(y_true)) * np.abs((y_pred - y_true)))
    return MAE

def SR(y_true,y_pred):
    SR=np.sum(y_true-y_pred)
    return SR'''

def Recall(y_true,y_pred):
    true_positive = np.sum(np.less(y_true, 0.2)* np.less(y_pred, 0.2))
    false_negative = np.sum(np.less(y_true, 0.2)*np.greater_equal(y_pred, 0.2))
    recall = true_positive / (true_positive + false_negative)
    return recall

def quantile_error_nan(y_true,y_pred,q):
    e=y_true-y_pred
    return np.nanmean(np.maximum(q*e,(q-1)*e))

def quantile_error(y_true,y_pred,q):
    e=y_true-y_pred
    return np.mean(np.maximum(q*e,(q-1)*e))

def power_law_predict(x, C0, t):
    n = x[1]
    k = x[0]
    y_pred = (C0 ** (1 - n) + (n - 1) * k * t) ** (1 / (1 - n))
    if len(y_pred)>1:
        y_pred[np.argwhere(y_pred < 0)] = 0
        y_pred[np.argwhere((C0 ** (1 - n) + (n - 1) * k * t) < 0)] = 0
    return y_pred

def first_order_predict(x,C0,t):
    k=x
    y_pred=C0*np.exp(-1*k*t)
    y_pred[np.argwhere(y_pred < 0)] = 0
    return y_pred

def Feben_Taras_predict(x, C0, t):
    n = x[1]
    k = x[0]
    y_pred=C0-k*t**n
    y_pred[np.argwhere(y_pred < 0)] = 0
    y_pred[np.argwhere(C0 < k*t**n)] = 0
    return y_pred

def limited_first_order_predict(x,C0,t):
    Cx = x[1]
    k = x[0]
    y_pred = Cx+(C0-Cx) * np.exp(-1 * k * t)
    if len(y_pred)>1:
        y_pred[np.argwhere(y_pred < 0)] = 0
    return y_pred

def limited_power_law_predict(x,C0,t):
    Cx= x[2]
    n = x[1]
    k = x[0]
    y_pred = Cx + (k*t*(n-1)+(C0-Cx)**(1-n))**(1/(1-n))
    if len(y_pred) > 1:
        y_pred[np.argwhere(y_pred < 0)] = 0
        y_pred[np.argwhere((C0-Cx)<0)]=0
        y_pred[np.argwhere((k*t*(n-1)+(C0-Cx)**(1-n)) < 0)] = 0
    return y_pred

def parallel_first_order_predict(x,C0,t):
    k2=x[2]
    k1=x[1]
    w=x[0]
    y_pred=w*C0* np.exp(-1 * k1 * t)+(1-w)*C0* np.exp(-1 * k2 * t)
    if len(y_pred) > 1:
        y_pred[np.argwhere(y_pred < 0)] = 0
    return y_pred

def limited_parallel_first_order_predict(x,C0,t):
    Cx=x[3]
    k2=x[2]
    k1=x[1]
    w=x[0]
    y_pred=Cx+w*(C0-Cx)* np.exp(-1 * k1 * t)+(1-w)*(C0-Cx)* np.exp(-1 * k2 * t)
    y_pred[np.argwhere(y_pred < 0)] = 0
    return y_pred

def parallel_power_decay_predict(x,C0,t):
    n2 = x[4]
    k2 = x[3]
    n1 = x[2]
    k1 = x[1]
    w = x[0]
    y_pred = ((w*C0) ** (1 - n1) + (n1 - 1) * k1 * t) ** (1 / (1 - n1))+(((1-w)*C0) ** (1 - n2) + (n2 - 1) * k2 * t) ** (1 / (1 - n2))
    y_pred[np.argwhere(y_pred < 0)] = 0
    y_pred[np.argwhere((w*C0 ** (1 - n1) + (n1 - 1) * k1 * t) < 0)] = 0
    y_pred[np.argwhere(((1-w) * C0 ** (1 - n2) + (n2 - 1) * k2 * t) < 0)] = 0
    return y_pred

def limited_parallel_power_decay_predict(x,C0,t):
    Cx=x[5]
    n2 = x[4]
    k2 = x[3]
    n1 = x[2]
    k1 = x[1]
    w = x[0]
    y_pred = Cx+((w*(C0-Cx)) ** (1 - n1) + (n1 - 1) * k1 * t) ** (1 / (1 - n1))+(((1-w)*(C0-Cx)) ** (1 - n2) + (n2 - 1) * k2 * t) ** (1 / (1 - n2))
    y_pred[np.argwhere(y_pred < 0)] = 0
    y_pred[np.argwhere((w*(C0-Cx) ** (1 - n1) + (n1 - 1) * k1 * t) < 0)] = 0
    y_pred[np.argwhere(((1-w) * (C0-Cx) ** (1 - n2) + (n2 - 1) * k2 * t) < 0)] = 0
    y_pred[np.argwhere((C0 - Cx) < 0)] = 0
    return y_pred





