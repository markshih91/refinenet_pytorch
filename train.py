import os
import sys
import time
import utils
import torch as t
from config import DefaultCofig as cfg
from net.refinenet.blocks import MyLoss
from net.refinenet.refinenet_4cascade import RefineNet4Cascade
from torch.utils import data
import dataset

# prefix_name
prefix_name = '_' + time.strftime('%m%d_%H:%M:%S')

# load net
net = RefineNet4Cascade(input_shape=(3, 160, 160), pretrained=False, num_classes=40)
if cfg.use_gpu:
    net.cuda()

# data preparation
train_data = dataset.NYUDV2Dataset(cfg.images, cfg.labels, cfg.depths, cfg.train_split)
train_dataLoader = data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)

val_data = dataset.NYUDV2Dataset(cfg.images, cfg.labels, cfg.depths, cfg.test_split)
val_dataLoader = data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)

optimizer = t.optim.Adam(net.parameters(), lr=cfg.lr)
criterion = MyLoss()
if cfg.use_gpu:
    criterion.cuda()

# train
best_loss = sys.maxsize
print('Train on {0} samples, validate on {1} samples'.format(train_data.__len__(), val_data.__len__()))
for epoch in range(cfg.epochs):

    print('Epoch {0}/{1}\n'.format(epoch + 1, cfg.epochs))

    epoch_start = time.time()

    for i, (x, y1, y2) in enumerate(train_dataLoader):

        step_start = time.time()

        if cfg.use_gpu:
            x = x.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()

        optimizer.zero_grad()
        y1_, y2_ = net(x)
        loss = criterion(y1_, y2_, y1, y2)

        loss.backward()
        optimizer.step()

        cur = time.time()

        print('{0:10d}/{1} {2} - USED: {3} - loss: {4:.4f}'
              .format((i + 1) * cfg.batch_size,
                      train_data.__len__(),
                      utils.progress_bar((i + 1) * cfg.batch_size / train_data.__len__()),
                      utils.eta_format(cur - epoch_start),
                      loss))

        t.cuda.empty_cache()

    # net.eval()

    total_loss = 0.0

    with t.no_grad():
        for i, (x, y1, y2) in enumerate(val_dataLoader):

            if cfg.use_gpu:
                x = x.cuda()
                y1 = y1.cuda()
                y2 = y2.cuda()

            y1_, y2_ = net(x)

            cur_loss = criterion(y1_, y2_, y1, y2).item()
            print(cur_loss)
            total_loss = total_loss + cur_loss

    mean_loss = total_loss / val_data.__len__()

    print('validation loss: {0}'.format(mean_loss))

    if mean_loss < best_loss:
        best_loss = mean_loss
        saved_model_name = os.path.join(cfg.saved_model_path,
                                        cfg.model + prefix_name + '.pkl')
        t.save(net.state_dict(), saved_model_name)
        print('saved model: ', saved_model_name)

    # save the newest model in current epoch
    saved_model_epoch_name = os.path.join(cfg.saved_model_path,
                                          cfg.model + prefix_name + '_epoch_{0}.pkl').format(epoch + 1)
    t.save(net.state_dict(), saved_model_epoch_name)
    print('saved model: ', saved_model_epoch_name)

    pre_saved_model_epoch_name = os.path.join(cfg.saved_model_path,
                                          cfg.model + prefix_name + '_epoch_{0}.pkl').format(epoch)
    if os.path.exists(pre_saved_model_epoch_name):
        os.remove(pre_saved_model_epoch_name)

    print()

    # net.train()
