# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from trainers.pix2pix_trainer import Pix2PixTrainer
import subprocess

import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import sys
from util.visualizer import Visualizer
import ipdb
import tqdm
import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

#torch.manual_seed(0)
# load the dataset
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create trainer for our model
trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)

# create tool for visualization
visualizer = Visualizer(opt)

# save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), opt.name)
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        # print(data_i['image'])
        iter_counter.record_one_iteration()


        #use for Domain adaptation loss
        p = min(float(i + (epoch - 1) * len_dataloader) / 50 / len_dataloader, 1)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # Training

        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, alpha=alpha)
        
        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(
                losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image',
                                    trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(
                visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

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
