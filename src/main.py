import os
import torch
import shutil
from torch.optim import Adam
from torch.nn import MSELoss
from torch import nn
from tqdm import tqdm
from data_loader import get_dataloader
from model import UOD
import matplotlib.pyplot as plt
import numpy as np
#the macro for the label to id

args = {
    
    'train_split': 0.98,
    'val_split': 0.02,
        
    "epochs": 100,
    "batch_size":64,
    "resume":False,
    "checkpoint_path": None,
    "checkpoint_every":2,
    "validate_every":2,
    "visualize_every":1,
    "lr": 0.0001,
    "seed": 354,
    "device":"cuda:0",
    'aux-loss': False,
    "variable-slots": False,
    "variable-init":False,

    'dataset':'object-room',
    'data':{
        'multi-object': './../data/clever',
        'object-room':'./../data/object-room'
    },
    
    "experiment_name":"more-data-simple-loss",
    "experiment_root":"../experiments",
    
    "hidden_dim": 64, 

    "image":{
        "size":64,
        "channels": 3
    },

    "encoder":{
        "out_channels": -1,
        "in_channels": -1,
    },

    "slot":{
        'number_of_slots': 7,
        'input_dim': -1,
        'slot_dim':-1,
        'hidden_dim':-1,
        'hidden_dimension_kv':-1,
        'iterations': 3,
        'eps': 10e-8,
        'gru': {'hidden_dim':64}
    },

    "decoder":{
        "in_channels": -1,
        "out_channels": -1,
        "hidden_channels": -1,
        "initial_resolution": (4,4)
    }
}

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if(m.bias is not None):
            nn.init.zeros_(m.bias)
        


class Trainer:

    def __init__(self, args):

        #add the arguments
        self.args = args
        
        #clean the args
        self.clean_args()

        #meta data
        self.resume = self.args['resume']
        self.epoch_no = 0
        self.score = 100
        self.epochs = self.args['epochs']
        self.device = self.args['device']
        self.checkpoint_every = self.args['checkpoint_every']
        self.validate_every = self.args['validate_every']
        self.visualize_every = self.args['visualize_every']
        self.expermiment_name = self.args['experiment_name']
        self.expermiment_name = self.args['dataset'] +"-" + self.expermiment_name
        self.checkpoint_path = self.args['checkpoint_path']

        #the dirs
        self.expermiment_dir = os.path.join(args['experiment_root'], self.expermiment_name)
        self.checkpoint_dir = os.path.join(self.expermiment_dir, 'checkpoints')
        self.visualization_dir = os.path.join(self.expermiment_dir, 'visualization')
        self.train_loss_file = os.path.join(self.expermiment_dir, 'train_loss.csv')
        self.val_loss_file = os.path.join(self.expermiment_dir, 'val_loss.csv')
        self.init_dirs()
        self.copy_code()

        #the model
        self.model = UOD(self.args)
        self.model.apply(weights_init)

        #restore the model
        if(self.args['resume']):
            self.restore()

        #move the model to designated device
        self.model.to(self.device)

        #get the datasets
        self.train_data, self.val_data = get_dataloader(self.args)

        #we will create a dataloader for each val dataset
        #the optimizer
        self.optimizer = Adam(self.model.parameters(), self.args['lr'])
        self.mse = MSELoss()

        #meta data
        self.num_of_slots = args['slot']['number_of_slots']
        self.lam = 0.001
    
    #clean the arguments
    def clean_args(self):
        hd = args['hidden_dim']
        
        args['slot']['input_dim'] = hd
        args['slot']['slot_dim'] = hd
        args['slot']['hidden_dim'] = hd
        args['slot']['hidden_dimension_kv'] = hd

        args['encoder']["in_channels"] = args['image']['channels']
        args['encoder']["out_channels"] = args['slot']['slot_dim']
        args['encoder']["hidden_channels"] = args['encoder']["out_channels"]
        
        args['decoder']["in_channels"] = args['slot']['slot_dim']
        args['decoder']["out_channels"] = args['image']['channels'] + 1
        args['decoder']["hidden_channels"] = args['slot']['slot_dim']
        
    #restore the model
    def restore(self):
        model_path = ''
        if(self.checkpoint_path is None):
            checkpoints = os.listdir(self.checkpoint_dir)
            if(len(checkpoints) > 0):
                checkpoints.sort(key= lambda x: int(x.split('.')[0].split('_')[1]))
                model_path = os.path.join(self.checkpoint_dir , checkpoints[-1])
        else:
            model_path = self.checkpoint_path
        if(model_path != ''):
            self.model.load_state_dict(torch.load(model_path))

    #create dirs
    def init_dirs(self):
        if(not os.path.exists(self.expermiment_dir)):
            os.makedirs(self.expermiment_dir)
        if(not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        if(not os.path.exists(self.visualization_dir)):
            os.makedirs(self.visualization_dir)
        if(not os.path.exists(self.train_loss_file) or not self.resume):
            with open(self.train_loss_file, "w") as f:
                pass
        if(not os.path.exists(self.val_loss_file) or not self.resume):
            with open(self.val_loss_file, "w") as f:
                pass
    
    def get_image(self, x, masks = False):
        if (not masks):
            x = torch.permute(x, (1, 2, 0))
        x = x.detach().to('cpu').numpy()
        x = x + np.min(x)
        x = x/np.max(x)
        return x

    def plot_results(self, results, X):
        
        #create folder for epcoch
        folder_path = os.path.join(self.visualization_dir, f'{self.epoch_no}')
        if(not os.path.exists(folder_path)):
            os.makedirs(folder_path)
        
        #get the data
        orig_images = X['images']
        orig_masks = X['masks']
        pred_images = results['reconstructed_image']
        pred_masks = results['object_masks']


        #iterate over all the batches
        for i in range(min(orig_images.shape[0],5)):
          
            #create image folder
            image_folder = os.path.join(folder_path, f'{i}')
            if(not os.path.exists(image_folder)):
                os.makedirs(image_folder)

            #get the images
            orig_image_i = self.get_image(orig_images[i])
            pred_image_i = self.get_image(pred_images[i])
            
            #plot the images
            fig, axes = plt.subplots(1,3)
            axes[0].imshow(orig_image_i)
            axes[0].set_title("orig image")
            axes[1].imshow(pred_image_i)
            axes[1].set_title("pred image")
            axes[2].imshow(orig_image_i - pred_image_i)
            axes[2].set_title("diff")
            plt.savefig(os.path.join(image_folder, 'orig.png'))

            #number of slots
            fig, axes = plt.subplots(2,self.num_of_slots)
            for j in range(self.num_of_slots):
                orig_mask_i = self.get_image(orig_masks[i][j], masks = True)
                pred_mask_i = self.get_image(pred_masks[i][j][0], masks = True)

                for axes_list in axes:
                    for ax in axes_list:
                        ax.set_xticks([])
                        ax.set_yticks([])
                
                axes[0][j].imshow(orig_mask_i)
                axes[0][j].set_title(f"o{j}")
                axes[1][j].imshow(pred_mask_i)
                axes[1][j].set_title(f"p{j}")
                
            plt.savefig(os.path.join(image_folder, 'masks.png'))
            plt.close()

    #visualize the results
    def visualize(self):
        
        #set in the eval mode
        self.model.eval()
        batches = 0

        #the return dict
        results = dict()

        with torch.no_grad():
            for X in self.val_data:

                #get the outcome
                results = self.model(X['images'])
                self.plot_results(results, X)
                break
        
    #copy code
    def copy_code(self):
        '''
        Copies code from src folder to the experiment folder
        '''
        code_path = os.path.join(self.expermiment_dir, 'src')
        shutil.rmtree(code_path, ignore_errors=True)
        shutil.copytree('../src/', code_path)

    #save the model
    def save(self, mode = 'train'):
        if(mode == 'train'):
            model_path = os.path.join(self.checkpoint_dir, f'model_{self.epoch_no}.pt')
        else:
            model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')    
        torch.save(self.model.state_dict(), model_path)
    
    #print the results
    def print_results(self, results):
        print(f'---EPOCH {self.epoch_no} ----')
        print(f'loss: {results}')
        print()
        print("-------")

    #saving results
    def save_results(self, results, file):
        with open(file, "a+") as f:
            for k in results:
                f.write(f"{results[k]},")
            f.write("\n")

    def auxilary_loss(self, masks = None):
        total_loss = 0
        if(masks is None):
            return total_loss
        for i in range(masks.shape[1]):
            for j in range(i+1, masks.shape[1]):
                total_loss += torch.sum(masks[:,i]*masks[:,j])/(128*128)
        return total_loss

    #losses
    def loss(self, Y_pred, Y_true, masks = None):
        results = dict()
        loss = self.mse(Y_pred, Y_true)
        results['loss'] = loss
        aloss = 0
        if(self.args['aux-loss']):
            aloss = self.auxilary_loss(masks)
            results['aloss'] = aloss
        results['tloss'] = loss + self.lam*aloss
        return results

    #the validator
    def validate(self):
        
        #set in the eval mode
        self.model.eval()
        total_loss = []
        batches = 0

        #the return dict
        results = dict()

        with torch.no_grad():
            for X in self.val_data:

                #get the outcome
                results = self.model(X['images'])

                #get the cross entropy loss
                loss = self.loss(results['reconstructed_image'], X['images'], results['object_masks'])
            
                total_loss.append(loss)
                batches += 1
        
            #get the final results
            total_loss = self.reduce_loss(total_loss, batches)
            return total_loss

    def reduce_loss(self, l, n):
        results = dict()
        for k in l[0]:
            results[k] = 0
        for i in l:
            for k in i:
                results[k] += i[k]/n
        return results

    def train_epoch(self):
        '''
        Trains the model for one epoch
        Returns:
            dictionary: {loss: <epoch loss>, }
        '''
        total_loss = []
        batches = 0

        #set the model in training mode
        self.model.train()

        #the results
        results = dict()

        #get the data
        '''
        data should be list of tensors 
        '''
        for X in tqdm(self.train_data):

            #can ignore X_lang for now
            #zeros the grad
            self.optimizer.zero_grad()

            #get the outcome
            results = self.model(X['images'])

            #get the cross entropy loss
            loss = self.loss(results['reconstructed_image'], X['images'], results['object_masks'])

            #backward
            loss['tloss'].backward()

            #optimizers step
            self.optimizer.step()

            total_loss.append(loss)
            batches += 1
        
        total_loss = self.reduce_loss(total_loss, batches)
        return total_loss

    def train(self):
        
        #print the message
        print('training started')

        #just train for all epochs
        for epoch in tqdm(range(1, self.epochs + 1)):

            #update epoch
            self.epoch_no = epoch

            #train for one epoch
            results = self.train_epoch()

            #print the results
            self.print_results(results)

            #save the results
            self.save_results(results, self.train_loss_file)

            #see for checkpointing
            if(epoch % self.checkpoint_every == 0):
                self.save()
            
            #do we eval?
            if(epoch % self.validate_every == 0):
                
                #get the results
                results = self.validate()
                self.print_results(results)
                self.save_results(results, self.val_loss_file)
                
                #save the best model
                if(self.score > results['tloss']):
                    self.score = results['tloss']
                    self.save(mode = 'best')

            #do we visualize?
            if(epoch % self.visualize_every == 0):
                self.visualize()

def main():
    trainer = Trainer(args)
    trainer.train()

main()