import numpy as np


def make_one_hot2d(data, cls_num):
    return (np.arange(cls_num) == data[:, :, None]).astype(np.integer).transpose(2, 0, 1)


def depth_transfer(depth_arr, max_value):
    return np.asarray(depth_arr * max_value, dtype=np.int8)


def seg_transfer(seg_arr):
    return np.argmax(seg_arr, axis=0) + 1


def progress_bar(rate, length=30):
    cur_len = int(rate * length)
    if cur_len == 0:
        bar = '[..............................]'
    elif cur_len < length:
        bar = '[' + ('=' * cur_len) + '>' + ('.' * (length - cur_len)) + ']'
    else:
        bar = '[==============================]'
    return bar


def eta_format(eta):
    if eta > 3600:
        res = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
    elif eta > 60:
        res = '%d:%02d' % (eta // 60, eta % 60)
    else:
        res = '%ds' % eta
    return res