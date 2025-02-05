import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
# from captum.attr import IntegratedGradients

import random

# from scipy.sparse import lil_matrix, csc_matrix
# from scipy.sparse.linalg import spsolve
import math

from resnet import resnet18,resnet50,ResNet9


random_list = [0,0.2,0.4,0.6,0.8,1.0]

sparsity = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


neighbors_weights = [((1, 1), 1 / 12), ((0, 1), 1 / 6), ((-1, 1), 1 / 12), ((1, -1), 1 / 12), ((0, -1), 1 / 6),
                     ((-1, -1), 1 / 12), ((1, 0), 1 / 6), ((-1, 0), 1 / 6)]

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

test_dataset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False)
glob_interval = 32*32 / 100

def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=True,
        use_tf32=False,
        max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        print("SEED ", seed)
        seed += 1
        np.random.seed(seed)
        print("SEED ", seed)
        seed += 1
        torch.manual_seed(seed)
        print("SEED ", seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
                print("SEED ", seed)

    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def ori_MoRF(net, inputs, targets, scores=None):
    acc_lists = []
    output = net(inputs.to(device))
    output = output.argmax().item()  # int(output.item()>threshold)
    if math.fabs(output - targets) < 0.5:
        acc_lists.append(1)
    else:
        acc_lists.append(0)

    interval = glob_interval  # scores.shape[0] // 50

    # score_tensor = torch.from_numpy(1-scores)

    score_vector = (1 - scores).reshape(-1)
    score_indexes = np.argsort(score_vector)  # min is the first

    for i in sparsity:
        masks = np.ones_like(score_vector)
        masks[score_indexes[:min(int(interval * (i + 1)), scores.shape[0])]] = 0.0
        masks = torch.from_numpy(masks).view(1, 1, inputs.shape[-2], inputs.shape[-1])

        output = net((inputs * masks).float().to(device))
        output = output.argmax().item()
        if math.fabs(output - targets) < 0.5:
            acc_lists.append(1)
        else:
            acc_lists.append(0)
    return acc_lists


def ori_LeRF(net, inputs, targets, scores=None):
    acc_lists = []
    output = net(inputs.to(device))
    output = output.argmax().item()  # int(output.item()>threshold)
    if math.fabs(output - targets) < 0.5:
        acc_lists.append(1)
    else:
        acc_lists.append(0)

    interval = glob_interval
    score_vector = scores.reshape(-1)
    score_indexes = np.argsort(score_vector)
    # for i in range(0,50):
    for i in sparsity:

        masks = np.ones_like(score_vector)
        masks[score_indexes[:min(int(interval * (i + 1)), scores.shape[0])]] = 0.0
        masks = torch.from_numpy(masks).view(1, 1, inputs.shape[-2], inputs.shape[-1])

        output = net((inputs * masks).float().to(device))
        output = output.argmax().item()
        if math.fabs(output - targets) < 0.5:
            acc_lists.append(1)
        else:
            acc_lists.append(0)
    return acc_lists


def ours_random_MoRF(net, inputs, targets,
                     robust_ratio=0.1, num_samples=50, max_ratio=0.1, use_max=False,
                     scores=None):
    acc_lists = []
    output = net(inputs.to(device))
    output = output.argmax().item()  # int(output.item()>threshold)
    if math.fabs(output - targets) < 0.5:
        acc_lists.append(1)
    else:
        acc_lists.append(0)

    def samples(input, masks, pexils_ids,robust_ratio):
        union_masks = torch.rand([num_samples, masks.shape[-3], masks.shape[-2], masks.shape[-1]],device=device)
        union_masks = torch.where(union_masks > robust_ratio, torch.ones_like(union_masks),
                                  torch.zeros_like(union_masks))
        new_masks = masks + (1 - masks) * union_masks.to(device)

        new_inputs = torch.tile(input, [num_samples, 1, 1, 1]).to(device)
        new_inputs = new_inputs * new_masks

        new_outputs = net(new_inputs.float())
        new_outputs = new_outputs.argmax(
            dim=-1).cpu()  

        results = new_outputs.float() - targets
        results = torch.where(torch.abs(results) < 0.5, torch.ones_like(results), torch.zeros_like(results))

        return results.mean()

    score_vector = (1 - scores).reshape(-1)
    score_indexes = np.argsort(score_vector) 

    all_pxiels = score_vector.shape[0]
    interval = glob_interval  
    for i in sparsity:
        masks = np.ones_like(score_vector)
        expl_size = min(int(interval * (i + 1)), scores.shape[0])
        masks[score_indexes[:expl_size]] = 0.0
        masks = torch.from_numpy(masks).view(1, 1, inputs.shape[-2], inputs.shape[-1]).to(device)

        ratio = robust_ratio
        if use_max:
            if expl_size*robust_ratio> max_ratio * all_pxiels:
                ratio = max_ratio * all_pxiels/ expl_size

        output = samples(inputs, masks, i,ratio)
        acc_lists.append(output.item())

    return acc_lists


def ours_random_LeRF(net, inputs, targets,
                     robust_ratio=0.1, num_samples=50, max_ratio=0.1, use_max=False,
                     scores=None):
    acc_lists = []
    output = net(inputs.to(device))
    output = output.argmax().item() 
    if math.fabs(output - targets) < 0.5:
        acc_lists.append(1)
    else:
        acc_lists.append(0)

    def samples(input, masks, pexils_ids,robust_ratio):
        union_masks = torch.rand([num_samples, masks.shape[-3], masks.shape[-2], masks.shape[-1]],device=device)
        union_masks = torch.where(union_masks > robust_ratio, torch.ones_like(union_masks),
                                  torch.zeros_like(union_masks))

        new_masks = masks + (1 - masks) * union_masks.to(device)

        new_inputs = torch.tile(input, [num_samples, 1, 1, 1]).to(device)
        new_inputs = new_inputs * new_masks

        new_outputs = net(new_inputs.float())
        new_outputs = new_outputs.argmax(
            dim=-1).cpu()  
        results = new_outputs.float() - targets
        results = torch.where(torch.abs(results) < 0.5, torch.ones_like(results), torch.zeros_like(results))

        return results.mean()

    score_vector = (scores).reshape(-1)
    score_indexes = np.argsort(score_vector)  

    interval = glob_interval  
    all_pxiels = score_vector.shape[0]
    for i in sparsity:
        masks = np.ones_like(score_vector)
        expl_size = min(int(interval * (i + 1)), scores.shape[0])
        masks[score_indexes[:expl_size]] = 0.0
        masks = torch.from_numpy(masks).view(1, 1, inputs.shape[-2], inputs.shape[-1]).to(device)

        ratio = robust_ratio
        if use_max:
            if expl_size*robust_ratio> max_ratio * all_pxiels:
                ratio = max_ratio * all_pxiels/ expl_size

        output = samples(inputs, masks, i,ratio)
        acc_lists.append(output.item())

    return acc_lists


global_dir = '../../data/cifar100/IG/'
explanation_name_list= [global_dir+'resnet-9-explanations_random0.000000_%d.npy',
                        global_dir+'resnet-9-explanations_random0.200000_%d.npy',
                        global_dir + 'resnet-9-explanations_random0.400000_%d.npy',
                        global_dir + 'resnet-9-explanations_random0.600000_%d.npy',
                        global_dir + 'resnet-9-explanations_random0.800000_%d.npy',
                        global_dir + 'resnet-9-explanations_random1.000000_%d.npy']

finetune_explanation_name_list= [global_dir+'resnet-9-finetune-explanations_random0.000000_%d.npy',
                        global_dir+'resnet-9-finetune-explanations_random0.200000_%d.npy',
                        global_dir + 'resnet-9-finetune-explanations_random0.400000_%d.npy',
                        global_dir + 'resnet-9-finetune-explanations_random0.600000_%d.npy',
                        global_dir + 'resnet-9-finetune-explanations_random0.800000_%d.npy',
                        global_dir + 'resnet-9-finetune-explanations_random1.000000_%d.npy']

def evaluate_ori_ori(seed):
    model = ResNet9(3, 100)
    state_dict = torch.load(global_dir + 'resnet-9.pth',map_location='cpu')
    model.load_state_dict(state_dict['net'])
    model.to(device)
    model.eval()
    net = model

    for explantion_name in explanation_name_list:

        path = explantion_name%seed
        if not os.path.exists(path):
            continue

        save_path = path.replace('.npy','_ori_model_ori.npy')
        if os.path.exists(save_path):
            continue

        explaination_list = np.load(path)
        acc_lists_LeRF = []
        acc_lists_MoRF = []
        count = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            scores = explaination_list[count].copy()
            scores = scores.reshape(-1)

            true_or_false = ori_LeRF(net, inputs.clone(), targets.clone(), scores=scores)
            acc_lists_LeRF.append(true_or_false)
            true_or_false = ori_MoRF(net, inputs.clone(), targets.clone(), scores=scores)
            acc_lists_MoRF.append(true_or_false)
            count += 1

        acc_lists_LeRF = np.array(acc_lists_LeRF)
        acc_lists_MoRF = np.array(acc_lists_MoRF)
        acc_LeRF = acc_lists_LeRF.mean(axis=0)
        acc_MoRF = acc_lists_MoRF.mean(axis=0)
        print(acc_LeRF)
        print(acc_MoRF)
        np.save( save_path,[acc_LeRF, acc_MoRF,acc_lists_LeRF,acc_lists_MoRF])

def evaluation_ori_rfid(seed, robust_ratio_MoRF=0.5, robust_ratio_LeRF=0.5):
    model = ResNet9(3, 100)
    state_dict = torch.load(global_dir + 'resnet-9.pth', map_location='cpu')
    model.load_state_dict(state_dict['net'])
    model.to(device)
    model.eval()
    net = model

    for explantion_name in explanation_name_list:

        path = explantion_name % seed
        if not os.path.exists(path):
            continue

        save_path = path.replace('.npy', '_ori_model_rfid.npy')
        if os.path.exists(save_path):
            continue

        explaination_list = np.load(path)


        # point evaluation
        acc_lists_LeRF = []
        acc_lists_MoRF = []
        count = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            scores = explaination_list[count].copy()
            scores = scores.reshape(-1)


            true_or_false = ours_random_LeRF(net, inputs.clone(), targets.clone(),
                                             robust_ratio=robust_ratio_LeRF,use_max=False,
                                             scores=scores)
            acc_lists_LeRF.append(true_or_false)
            true_or_false = ours_random_MoRF(net, inputs.clone(), targets.clone(),
                                             robust_ratio=robust_ratio_MoRF,use_max=False,
                                             scores=scores)
            acc_lists_MoRF.append(true_or_false)
            count += 1

        acc_lists_LeRF = np.array(acc_lists_LeRF)
        acc_lists_MoRF = np.array(acc_lists_MoRF)
        acc_LeRF = acc_lists_LeRF.mean(axis=0)
        acc_MoRF = acc_lists_MoRF.mean(axis=0)
        print(acc_LeRF)
        print(acc_MoRF)
        np.save(
            save_path,[acc_LeRF,acc_MoRF,acc_lists_LeRF,acc_lists_MoRF])

def evaluation_finetune_ffid(seed, robust_ratio_MoRF=0.5, robust_ratio_LeRF=0.5,threshold=0.1):
    model = ResNet9(3, 100)
    state_dict = torch.load(global_dir + 'resnet-9-finetune.pth', map_location='cpu')
    model.load_state_dict(state_dict['net'])
    # model = resnet18_init
    model.to(device)
    model.eval()
    net = model

    for explantion_name in finetune_explanation_name_list:
        path = explantion_name % seed
        if not os.path.exists(path):
            continue

        save_path = path.replace('.npy', '_finetune_model_ffid.npy')
        if os.path.exists(save_path):
            continue

        explaination_list = np.load(path)

        # point evaluation
        acc_lists_LeRF = []
        acc_lists_MoRF = []
        count = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            scores = explaination_list[count].copy()
            scores = scores.reshape(-1)


            true_or_false = ours_random_LeRF(net, inputs.clone(), targets.clone(),
                                             robust_ratio=robust_ratio_LeRF,use_max=True,
                                             scores=scores)
            acc_lists_LeRF.append(true_or_false)
            true_or_false = ours_random_MoRF(net, inputs.clone(), targets.clone(),
                                             robust_ratio=robust_ratio_MoRF,use_max=True,
                                             scores=scores)
            acc_lists_MoRF.append(true_or_false)
            count += 1

        acc_lists_LeRF = np.array(acc_lists_LeRF)
        acc_lists_MoRF = np.array(acc_lists_MoRF)
        acc_LeRF = acc_lists_LeRF.mean(axis=0)
        acc_MoRF = acc_lists_MoRF.mean(axis=0)
        print(acc_LeRF)
        print(acc_MoRF)
        np.save(
            save_path,
            [acc_LeRF, acc_MoRF, acc_lists_LeRF, acc_lists_MoRF])
        
if __name__ == "__main__":
    for i in range(0,5):
        init_dl_program(device, seed=i)
        evaluate_ori_ori(i)
        evaluation_ori_rfid(i)
        evaluation_finetune_ffid(i)

