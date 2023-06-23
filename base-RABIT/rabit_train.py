# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import sys
from PIL import Image
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from trainers.pix2pix_trainer import Pix2PixTrainer
import subprocess
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import ipdb
import tqdm
import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1
jt.cudnn.set_max_workspace_ratio(0.0)

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create trainer for our model
trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    opt.epoch = epoch
    if not opt.maskmix:
        print('inject nothing')
    elif opt.maskmix and opt.noise_for_mask and epoch > opt.mask_epoch:
        print('inject noise')
    else:
         print('inject mask')
    print('real_reference_probability is :{}'.format(dataloader.dataset.real_reference_probability))
    print('hard_reference_probability is :{}'.format(dataloader.dataset.hard_reference_probability))
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        
        iter_counter.record_one_iteration()

        #use for Domain adaptation loss
        p = min(float(i + (epoch - 1) * len_dataloader) / 50 / len_dataloader, 1)
        alpha = 2. / (1. + np.exp(-10 * p, dtype="float32")) - 1

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, alpha=alpha)
        
        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            try:
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, 
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            except OSError as err:
                print(err)

        if iter_counter.needs_displaying():
            imgs_num = data_i['label'].shape[0]
            if opt.dataset_mode == 'celebahq':
                data_i['label'] = data_i['label'][:,::2,:,:]
            elif opt.dataset_mode == 'celebahqedge':
                data_i['label'] = data_i['label'][:,:1,:,:]
            elif opt.dataset_mode == 'deepfashion':
                data_i['label'] = data_i['label'][:,:3,:,:]
            elif opt.dataset_mode == 'ade20klayout':
                data_i['label'] = (data_i['label'][:,:3,:,:] -128) / 128
            elif opt.dataset_mode == 'cocolayout':
                data_i['label'] = (data_i['label'][:,:3,:,:] -128) / 128
            if data_i['label'].shape[1] == 3:
                label = data_i['label']
            else:
                label = data_i['label'].expand(-1, 3, -1, -1).float() / data_i['label'].max()

            imgs = jt.concat((label.cpu(), data_i['ref'].cpu(), trainer.out['warp128'].cpu(),
                                  trainer.get_latest_generated().data.cpu(), data_i['image'].cpu()), 0)

            im_sv_path = opt.checkpoints_dir.split('checkpoints')[0] + 'summary/' + opt.name
            if not os.path.exists(im_sv_path):
                os.makedirs(im_sv_path)
            try:
                jt.save_image(imgs, im_sv_path + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '.png',
                        nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        try:
            trainer.save('latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)

print('Training was successfully finished.')