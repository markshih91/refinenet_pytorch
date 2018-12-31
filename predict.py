import os
import time
from PIL import Image
import torch as t
from config import DefaultCofig as cfg
from net.refinenet.refinenet_4cascade import RefineNet4Cascade
from torch.utils import data
import utils
import dataset

load_start = time.time()

# load net
net = RefineNet4Cascade(input_shape=(3, 480, 640), num_classes=40)
if cfg.use_gpu:
    net.cuda()

net.load_state_dict(t.load(cfg.test_model))


predict_data = dataset.PredictINputDataset(cfg.predict_images)
predict_dataLoader = data.DataLoader(predict_data, batch_size=1)

load_end = time.time()
print('load model time: {0}ms'.format(1000 * (load_end - load_start)))

start = time.time()
with t.no_grad():
    for i, (x, name) in enumerate(predict_dataLoader):

        cur_start = time.time()

        if cfg.use_gpu:
            x = x.cuda()

        y_ = net(x)

        seg = Image.fromarray(seg.astype('uint8'))
        
        seg.save(os.path.join(cfg.predict_labels, name[0] + '.png'))

        cur_end = time.time()
        print('img {0} USED: {1}ms, predict total: {2}ms'.format(i + 1, 1000 * (cur_end - cur_start),
                      1000 * (cur_end - start)))