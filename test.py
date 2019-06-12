import torch as t
from config import DefaultCofig as cfg
from net.refinenet.blocks import MyLoss
from net.refinenet.refinenet_4cascade import RefineNet4Cascade
from torch.utils import data
import dataset

# load net
net = RefineNet4Cascade(input_shape=(3, 160, 160), num_classes=40)
if cfg.use_gpu:
    net.cuda()

net.load_state_dict(t.load(cfg.test_model))

# data preparation
test_data = dataset.NYUDV2Dataset(cfg.images, cfg.labels, cfg.depths, cfg.test_split)
test_dataLoader = data.DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True)

criterion = MyLoss()
if cfg.use_gpu:
    criterion.cuda()


total_loss = 0.0

with t.no_grad():
    for i, (x, y1, y2) in enumerate(test_dataLoader):

        if cfg.use_gpu:
            x = x.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()

        y1_, y2_ = net(x)

        cur_loss = criterion(y1_, y2_, y1, y2).item()
        print(cur_loss)
        total_loss = total_loss + cur_loss

mean_loss = total_loss / test_data.__len__()

print('validation loss: {0}'.format(mean_loss))
