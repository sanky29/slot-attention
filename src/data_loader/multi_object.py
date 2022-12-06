import h5py
import torch
from tqdm import tqdm
import numpy as np
import os
class MultiObject1:

    def __init__(self,args):
        
        #get args
        self.filename = args['data']['multi-object']
        self.device = args['device']
        self.batch_size = args['batch_size']

        self.number_of_files = 3
        self.data_len = 0
        self.files = {}

        #create file objects
        for i in range(self.number_of_files):
            path = os.path.join(self.filename, f'{i}.h5')
            self.files[i] = h5py.File(path, "r")
            self.data_len += self.files[i]['images'].shape[0]
        
        off = self.number_of_files*self.batch_size        
        self.batch_per_file = (self.data_len - off) // (self.batch_size * self.number_of_files)
        self.data_len = self.batch_per_file*self.number_of_files

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        #get the file
        file_id = index // self.batch_per_file
        data_id = index % self.batch_per_file
        
        si = data_id*self.batch_size
        en = si + self.batch_size
        
        
        images = torch.Tensor(self.files[file_id]['images'][si:en, ...])/255
        masks = torch.Tensor(self.files[file_id]['masks'][si:en, ...])

        return {"images": images, "masks": masks}         

class MultiObject2:

    def __init__(self,args):
        
        #get args
        self.filename = args['data']['multi-object']
        self.device = args['device']
        self.batch_size = args['batch_size']

        self.number_of_files = 4
        self.images = []
        self.masks = []

        #create file objects
        print('READING DATA')
        for i in tqdm(range(self.number_of_files)):
            path = os.path.join(self.filename, f'{i}.h5')
            file = h5py.File(path, "r")

            images = np.array(file['images'])
            images = torch.Tensor(images)
            images = images.float()
            images = images/255.0

            masks = np.array(file['masks'])
            masks = torch.Tensor(masks)
            
            self.images.append(images)
            self.masks.append(masks)

        self.images = torch.cat(self.images, dim = 0)
        self.masks = torch.cat(self.masks, dim = 0)

        self.data_len = self.images.shape[0]//self.batch_size

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        
        si = index*self.batch_size
        en = si + self.batch_size
        
        images = self.images[si:en,...]
        masks = self.masks[si:en,...]

        return {"images": images, "masks": masks}         