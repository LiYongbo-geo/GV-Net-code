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
from matplotlib import cm
from sympy import fraction
import torch
from torch.nn import functional as F
from matplotlib import rcParams
import datagen
import train
import utils
import decoder
from pyevtk.hl import imageToVTK,gridToVTK
import matplotlib
import encoder
import discriminator

#matplotlib.use("pgf")
pgf_config = {
    "font.family":'serif',
    "font.size": 18,
    "pgf.rcfonts": False,
    #"text.usetex": True,
    "pgf.preamble": [
        r"\usepackage{unicode-math}",
        #r"\setmathfont{XITS Math}", 
        # 这里注释掉了公式的XITS字体，可以自行修改
        r"\setmainfont{Times New Roman}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{CJKmath=true}",
        r"\setCJKmainfont{SimSun}",
    ],
}
rcParams.update(pgf_config)

def Model(m, w, filename = "density"):
    m = m.T
    L, W, H= m.shape
    c = ["#D1FEFE", "#D1FEFE", "#00FEF9", "#00FDFE", "#50FB7F", "#D3F821", "#FFDE00", "#FF9D00", "#F03A00", "#E10000"]
    x, y, z = np.indices((L, W, H))
    model = (x < 0) & (y < 0) & (z < 0)
    color = np.empty(m.shape, dtype=object)
    for i in range(L):
        for j in range(W):
            for k in range(H):
                if m[i][j][k] >= w and m[i][j][k] <=1:
                    cube = (x > i-1) & (x <= i)& (y > j-1) & (y <= j) & (z > k-1) & (z <= k)
                    color[cube] = c[int(round(10*m[i][j][k]))-1]
                    model = model | cube
    plt_model(model, color, filename)

def plt_model(model, facecolors, filename = "density"):
    fig = plt.figure(figsize = (12, 12))
    ax = fig.gca(projection='3d')
    ax.voxels(model, facecolors=facecolors)
    ticks = []
    for i in range(65):
        ticks.append("")
    ticks[0] = -16
    ticks[16] = -8
    ticks[32] = 0
    ticks[48] = 8
    ticks[64] = 16
    fontsize = 18
    ax.tick_params(pad = 10)
    plt.xticks(np.arange(0, 65, 1), ticks, fontsize = fontsize)
    ax.set_xlabel('Easting (Km)', labelpad=2)
    plt.yticks(np.arange(0, 65, 1), ticks, fontsize = fontsize)
    ax.set_ylabel('Northing (Km)', labelpad=2)
    
    zticks = []
    for i in range(33):
        zticks.append("")
    zticks[0] = 0
    zticks[8] = -3.2
    zticks[16] = -6.4
    zticks[24] = -9.6
    zticks[32] = -12.8
    
    ax.set_zticks(np.arange(0, 33, 1), zticks, fontsize = fontsize)
    ax.set_zlabel('Depth (Km)', labelpad=16)
    ax.invert_zaxis()
    ax.xaxis.set_tick_params(pad=-2)
    ax.yaxis.set_tick_params(pad=-2)
    ax.zaxis.set_tick_params(pad=10)
    path = "./figureout"
    pngpath = os.path.join(path, filename+".png")
    pdfpath = os.path.join(path, filename+".pdf")
    cb = plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(0,1), cmap=colorma()),
                      shrink=0.5, aspect=20, pad = 0.09)
    
    plt.savefig(pngpath)
    plt.savefig(pdfpath)
    #plt.show()

def plt_m(y, color='b'):
    x = range(1, len(y)+1)
    plt.xlim(-100, 16500)
    plt.ylim(0, 1.01)
    plt.xticks(np.arange(0, 16500, step=1024), fontsize=4)
    for i in range(len(y)):
        if y[i]==1:
            plt.scatter(x[i], y[i],linewidths=0.001)
    plt.plot(x, y, linewidth=0.8, color=color, linestyle=":")
    
    
def colorma():
    cdict = ["#F2F2F2", "#D1FEFE", "#00FEF9", "#00FDFE", "#50FB7F", "#D3F821", "#FFDE00", "#FF9D00", "#F03A00", "#E10000"] 
    return colors.ListedColormap(cdict, 'indexed')

def load_args():
    parser = argparse.ArgumentParser(description='gangravity')
    parser.add_argument('-c', '--checkpoint', default='./models/checkpoint_myloss.pt',
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
    density_modelone = np.zeros(shape = [32, 64 ,64])
    density_modelone[4:10, 22:42, 5:20] = 0.8
    density_modelone[8:12, 20:38, 35:55] = 0.8
    den_modelone = torch.from_numpy(density_modelone)
    den_modelone = den_modelone.unsqueeze(0)
    Forward = decoder.GravDecoder()
    field_modelone = Forward.forward(data_input = den_modelone)
    field_modelone = torch.sum(field_modelone,axis=1).squeeze(0)
    field_modelone = field_modelone.squeeze(0)
    print("1")

    """
    filename = "density_modelone"
    tru_model = density_modelone
    Model(tru_model, 0.01, filename = filename)

    plt.figure(figsize=(6, 6), dpi = 150)
    plt.imshow(field_modelone)
    plt.colorbar()
    plt.savefig("figureout/modelone_field.png")
    """

    print("2")
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

    with torch.no_grad():

        total_field = field_modelone.unsqueeze(0)
        total_field = total_field.unsqueeze(0)
        total_field = total_field.type(torch.FloatTensor)
        start_time = datetime.now()
        total_field =total_field.to(args_gen.device)

        density_rec = netEnc(total_field)
        end_time = datetime.now()
        print("预测耗时：", (end_time - start_time)/100)
        density = den_modelone.unsqueeze(0)
        density = density.to(args_gen.device)
        loss = my_loss(density_rec, density)
        metrics = torch.norm(density_rec - density)/(torch.norm(density_rec)+torch.norm(density))
        print("Mrteic:", metrics.item())
        tru_model = density_rec.squeeze(0).cpu().numpy()

        np.save("modeone_rec.npy", tru_model)