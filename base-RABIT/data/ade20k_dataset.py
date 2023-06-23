# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import json


class ADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default='',
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        print(f"label_dir: {label_dir}, image_dir: {opt.image_dir}, instance_dir: {opt.instance_dir}")
        # exit(0)
        if len(opt.image_dir) > 0:
            self.image_dir = opt.image_dir
            image_paths = make_dataset(
                self.image_dir, recursive=False, read_cache=True)
        else:
            image_paths = []

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(
                instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        if opt.isTrain:
            self.isTrain = True
            assert len(label_paths) == len(
                image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        else:
            self.isTrain = False
        # for line in f:
            # a = line.strip().split(',')
            # self.ref_dict[a[0]] = a[1]
        self.name = "ADE20KDataset"

        return label_paths, image_paths, instance_paths

    def get_ref(self, opt):
        # TODO: 对齐

        self.ref_dict = {}
        with open('./data/label_to_img.json', 'r') as f:
            self.ref_dict = json.load(f)
        
        ref_dict = self.ref_dict
        train_test_folder = ('training', 'validation')
        return ref_dict, train_test_folder
    
    def imgpath_to_labelpath(self, path):
        path = path.replace("img","label")
        return path


