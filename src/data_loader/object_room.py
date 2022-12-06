import h5py
import torch
from tqdm import tqdm
import numpy as np
import os
class ObjectRoom:

    def __init__(self,args):
        
        #get args
        self.filename = args['data']['object-room']
        self.device = args['device']
        self.batch_size = args['batch_size']

        self.number_of_files = 15
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