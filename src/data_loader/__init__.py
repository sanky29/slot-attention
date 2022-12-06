from .multi_object import MultiObject1, MultiObject2
from .object_room import ObjectRoom
from torch.utils.data import DataLoader
from torch.utils.data import random_split

__all__ = ['get_dataloader']
device = 'cpu'
def get_dataloader(args, mode = 'train'):
    global device
    data = None
    device = args['device']
    if(args['dataset'] == 'multi-object'):
        data = MultiObject2(args)
    if(args['dataset'] == 'object-room'):
        data = ObjectRoom(args)
    
    if(mode == 'train'):
        
        #get train size
        train_size = int(args['train_split']*len(data))
        
        #get val size
        val_size = len(data) - train_size
        
        #split the data
        splits = random_split(data, [train_size, val_size])

        #get data loaders
        batch_size = args['batch_size']
        train_dataloader = DataLoader(splits[0], batch_size = 1, shuffle = True, collate_fn = collate_fn)
        val_dataloader = DataLoader(splits[1], batch_size = 1, shuffle = True, collate_fn = collate_fn)

        return train_dataloader, val_dataloader
    
    else:
        test_dataloader =  DataLoader(data, batch_size = 1, shuffle = True, collate_fn = collate_fn)
        return test_dataloader


def collate_fn(x):
    x = x[0]
    for i in x:
        x[i] = x[i].to(device)
    return x