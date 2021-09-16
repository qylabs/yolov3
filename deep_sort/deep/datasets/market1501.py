# encoding: utf-8
"""
copy from  JDAI-CV/fast-reid/data/datasets/market1501.py
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re
import warnings
import shutil


class Market1501:
    """Market1501.
    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets',dataset_dir='', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        # data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if not osp.isdir(self.data_dir):
            raise ValueError("wrong data dir {}".format(self.data_dir))
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        
        # train = lambda: self.process_dir(self.train_dir)
        # query = lambda: self.process_dir(self.query_dir, is_train=False)
        # gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
        #                   (self.process_dir(self.extra_gallery_dir, is_train=False) if self.market1501_500k else [])

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            # if is_train:
            #     pid = self.dataset_name + "_" + str(pid)
            #     camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))
        return data
    
    def build_dataset(self,target_path,src_dir):
        if not osp.isdir(target_path):
            os.makedirs(target_path)
        
        src_data=self.process_dir(src_dir)
        for img,pid,_ in src_data:
            img_pid_path=osp.join(target_path,str(pid))
            if not osp.isdir(img_pid_path):
                os.makedirs(img_pid_path)
            shutil.copy(img,img_pid_path)

        print("Done! at ",target_path)

        



if __name__=="__main__":
    root_path='/home/wangtao/project/deep_sort_pytorch'
    dataset_dir='data/Market-1501-v15.09.15'
    target_dir=osp.join(root_path,'data/market1501')

    data_manager=Market1501(root_path,dataset_dir=dataset_dir)

    target_path_train=osp.join(target_dir,'train')
    data_manager.build_dataset(target_path_train,data_manager.train_dir)

    target_path_test=osp.join(target_dir,'test')
    data_manager.build_dataset(target_path_test,data_manager.query_dir)



