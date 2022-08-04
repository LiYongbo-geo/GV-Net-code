from datetime import datetime
import argparse
from fileinput import filename
import numpy as np
import pathlib
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import rcParams
import torch
from torch.nn import functional as F
from matplotlib import cm
import datagen
import train
import utils
from pyevtk.hl import imageToVTK,gridToVTK
import datagen
import encoder
import decoder
import discriminator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset,Dataset

def load_args():
    parser = argparse.ArgumentParser(description='gangravity')
    parser.add_argument('-c', '--checkpoint', default='./models/checkpoint16.pt',
                        type=str, help='checkpoint file')
    parser.add_argument('-d', '--device', default='cuda:0', type=str, help='computing device')
    parser.add_argument('-l', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-g', '--n_gp', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-n', '--n_batch', default=150, type=int)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-w', '--world_size', default=3, type=int)
    parser.add_argument('--use_spectral_norm', default=False)
    args = parser.parse_args()
    return args

def load_models(args, checkpoint=None):
    netDec = decoder.GravDecoder()
    netDis = discriminator.GravDiscriminator()
    netEnc = encoder.GravEncoder()
    netEnc = netEnc.to(args.device)
    #netDec = DDP(netDec.to(rank),device_ids=[rank])
    if checkpoint:
        netEnc.load_state_dict(checkpoint['enc_state_dict'])
        netDis.load_state_dict(checkpoint['dis_state_dict'])
#    netEnc = torch.nn.DataParallel(netEnc)
#    netDec = torch.nn.DataParallel(netDec)
#    netDis = torch.nn.DataParallel(netDis) 
    print (netDec, netDis, netEnc)
    return (netDec, netDis, netEnc)
    
def dice(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.view(num, -1)  
    m2 = target.view(num, -1)  
    intersection = m1 * m2
    loss = (2. * intersection.sum(1) + smooth) / ((m1*m1).sum(1) + (m2*m2).sum(1) + smooth)
    return loss.sum()/num
def my_loss(pre_y, tru_y): 
    loss = 1 - dice(pre_y, tru_y)
    return loss

if __name__ == '__main__':
    nzyx = (32,64,64)
    args_gen = load_args()
    checkpoint = torch.load(args_gen.checkpoint)
    saved_model = ['enc_state_dict','dis_state_dict']
    for model_key in saved_model:
        new_keys = list(map(lambda x:x[7:],checkpoint[model_key].keys()))
        checkpoint[model_key] = dict(zip(new_keys,list(checkpoint[model_key].values())))
    netDec,netDis,netEnc = load_models(args_gen, checkpoint)
    netDec.eval()
    netDis.eval()
    netEnc.eval()
    train_length = 100
    tes_loss  = 0
    tes_metrix = 0

    for index_train in  range(train_length):
        index_train = index_train*10 +1
        with torch.no_grad():

            gravtest = 'field{}.npy'.format(str(index_train))
            gravfile = os.path.join(os.getcwd(), 'traindatasets', "fields", gravtest)
            field = np.load(gravfile)
            total_field = torch.from_numpy(field).unsqueeze(0)
            total_field = total_field.unsqueeze(0)
            density_rec = netEnc(total_field)

            modelname = 'model{}.npy'.format(str(index_train))
            testfile = os.path.join(os.getcwd(), 'traindatasets', "models", modelname)
            density = np.load(testfile)
            density = torch.from_numpy(density).unsqueeze(0)
            density = density.to(args_gen.device)
            loss = my_loss(density_rec, density)
            tes_loss += loss.item()
            metrics = torch.norm(density_rec - density)/(torch.norm(density_rec)+torch.norm(density))
            tes_metrix += metrics.item()
            
            rec = density_rec.squeeze(0)
            rec = rec.cpu().numpy()
            rec = rec.reshape(64, 64)
            
            plt.figure(figsize=(5, 5), dpi = 150)
            plt.imshow(field, origin='lower')
            plt.title("Input")
            plt.colorbar()

            path = os.path.join("result", "{}.png".format(index_train))
            plt.savefig(path)