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
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel

from resnet import resnet18,resnet50,ResNet9

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from copy import deepcopy

import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

test_dataset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False)

trainset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(    trainset, batch_size=1, shuffle=False)


def explain_ori_model_cam():
    model = ResNet9(3, 100)
    state_dict = torch.load('../../data/cifar100/LayerCAM/resnet-9.pth',map_location='cpu')
    model.load_state_dict(state_dict['net'])
    model.to(device)
    model.eval()

    grad_cam = LayerCAM(model, [model.res2[-1]] ,use_cuda=True)
    # ScoreCAM()

    explanation_maps = []
    count = 0
    for inputs, labels in tqdm(val_loader):

        # inputs, labels = inputs.to(device), labels.to(device)
        # heatmap,idx = generate_gradcam(model,inputs)

        attributions = grad_cam(inputs, targets=[ClassifierOutputTarget(labels)])
        # attributions = grad_cam.attribute(inputs)
        # attributions = visualization.rescale(attributions, inputs.size)

        # attributions = torch.nn.functional.interpolate(attributions, size=(inputs.shape[-2], inputs.shape[-1]),
        #                                                mode='bilinear', align_corners=False)

        attributions = np.transpose(attributions, (1, 2, 0))
        # attributions = np.maximum(0, attributions)
        heatmap = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-10)

        # show the case :

        # image = inputs.cpu().detach().numpy()[0]
        # image = image.transpose([1,2,0])
        # image = image* np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        # image = np.clip(image,0,1)
        # plt.imsave('./image.jpg',image)

        # # cv
        # image = image[:,:,::-1]
        # cv2.imwrite('image.jpg',(image*255).astype(np.uint8))
        # heatmap_img = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        # super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, (image*255).astype(np.uint8), 0.5, 0)
        # cv2.imwrite('weighted_image.jpg',super_imposed_img)

        # denormalize
        # plt.imsave('./image.jpg',image)
        # plt.close()
        # plt.figure()
        # plt.imshow(heatmap[:,:,0], cmap='hot', interpolation='nearest')
        # plt.savefig('heatmap.jpg')


        # heatmap = grad_cam(model,inputs,labels)
        # heatmap = generate_gradcam_heatmap(model,labels,inputs)
        explanation_maps.append(heatmap)
    explanation_maps = np.stack(explanation_maps)
    np.save('../../data/cifar100/LayerCAM/resnet-9-explanations.npy' ,explanation_maps)

def explain_ori_model_IGSQ():
    model = ResNet9(3, 100)
    state_dict = torch.load('../../data/cifar100/IG/resnet-9.pth',map_location='cpu')
    model.load_state_dict(state_dict['net'])
    model.to(device)
    model.eval()
    
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)

    explanation_maps = []
    count = 0
    for inputs, labels in tqdm(val_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        # heatmap,idx = generate_gradcam(model,inputs)

        attributions = nt.attribute(inputs, n_samples=10, nt_type='smoothgrad_sq',
                                                    target=labels.to(torch.int))
        
        
        attributions = attributions.squeeze().cpu().detach().numpy()
        explanation_maps.append(attributions)

        # image = inputs.cpu().detach().numpy()[0]
        # image = image.transpose([1,2,0])
        # image = image* np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        # image = np.clip(image,0,1)
        # # plt.imsave('./image.jpg',image)
        # # # cv
        # attributions = attributions.sum(axis=0)
        # heatmap = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-10)

        # image = image[:,:,::-1]
        # cv2.imwrite(f'image_{count}_ori.jpg',(image*255).astype(np.uint8))
        # heatmap_img = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        # super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, (image*255).astype(np.uint8), 0.5, 0)
        # cv2.imwrite(f'weighted_image_{count}_ori.jpg',super_imposed_img)

        # count += 1

    explanation_maps = np.stack(explanation_maps)
    np.save('../../data/cifar100/IG/resnet-9-explanations.npy' ,explanation_maps)

def explain_ori_model_vis():
    model = ResNet9(3, 100)
    # resnet18_init = models.resnet18(pretrained=True)
    # num_ftrs = resnet18_init.fc.in_features
    # resnet18_init.fc = nn.Linear(num_ftrs, 100)
    state_dict = torch.load('../../data/cifar100/LayerCAM/resnet-9-finetune.pth',map_location='cpu')
    model.load_state_dict(state_dict['net'])
    model.to(device)
    model.eval()

    grad_cam = LayerCAM(model, [model.res2[-1]] ,use_cuda=True)
    # ScoreCAM()

    explanation_maps = []
    count = 0
    for inputs, labels in tqdm(val_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        # heatmap,idx = generate_gradcam(model,inputs)
        output = model(inputs)
        _,out_id = output.max(dim=-1)
        correct = out_id==labels

        attributions = grad_cam(inputs, targets=[ClassifierOutputTarget(labels)])
        # attributions = grad_cam.attribute(inputs)
        # attributions = visualization.rescale(attributions, inputs.size)

        # attributions = torch.nn.functional.interpolate(attributions, size=(inputs.shape[-2], inputs.shape[-1]),
        #                                                mode='bilinear', align_corners=False)

        attributions = np.transpose(attributions, (1, 2, 0))
        # attributions = np.maximum(0, attributions)
        heatmap = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-10)

        # show the case :

        image = inputs.cpu().detach().numpy()[0]
        image = image.transpose([1,2,0])
        image = image* np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        image = np.clip(image,0,1)
        # plt.imsave(f'./vis/image_{count}.jpg',image)

        # # cv
        image = image[:,:,::-1]
        cv2.imwrite(f'./vis1/{count}_image.jpg',(image*255).astype(np.uint8))
        heatmap_img = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, (image*255).astype(np.uint8), 0.5, 0)
        cv2.imwrite(f'./vis1/{count}_layercam_{correct.to(torch.long).item()}.jpg',super_imposed_img)

        # denormalize
        # plt.imsave('./image.jpg',image)
        # plt.close()
        # plt.figure()
        # plt.imshow(heatmap[:,:,0], cmap='hot', interpolation='nearest')
        # plt.savefig('heatmap.jpg')


        # heatmap = grad_cam(model,inputs,labels)
        # heatmap = generate_gradcam_heatmap(model,labels,inputs)
        count+=1
        if count > 100:
            break
    #     explanation_maps.append(heatmap)
    # explanation_maps = np.stack(explanation_maps)
    # np.save('../../data/cifar100/GradCAM/resnet-9-finetune-explanations.npy' ,explanation_maps)


def generate_explanations(random = 0.2):
    dir_path = '../../data/cifar100/LayerCAM/resnet-9-explanations.npy'
    explanations_list = np.load(dir_path)
    explanations_list = explanations_list[:,:,:,0] #sum(axis=1)

    if not os.path.exists(dir_path.replace('.npy','_random%f_%d.npy'%(0.0,0))):
        np.save(dir_path.replace('.npy','_random%f_%d.npy'%(0.0,0)),explanations_list)

    for seed in range(5):
        explanations = []
        for explanation in explanations_list:
            scores = deepcopy(explanation) #explaination_list[count].copy()
            ori_shape = scores.shape
            # scores = scores.sum(axis=0)
            scores = scores.reshape(-1)

            num = int(random * scores.shape[0])
            select = np.arange(scores.shape[0])
            idx_select = np.random.choice(select, num, replace=False)
            scores[idx_select] = np.random.permutation(scores[idx_select])

            explanations.append(scores.reshape(ori_shape))
        np.save(dir_path.replace('.npy','_random%f_%d.npy'%(random,seed)) , explanations)

def explain_train():
    model = ResNet9(3, 100)
    # num_ftrs = resnet18_init.fc.in_features
    # resnet18_init.fc = nn.Linear(num_ftrs, 100)
    state_dict = torch.load('../../data/cifar100/LayerCAM/resnet-9.pth',map_location='cpu')
    model.load_state_dict(state_dict['net'])
    model.to(device)
    model.eval()

    grad_cam = LayerCAM(model, [model.res2[-1]] ,use_cuda=True)

    explanations_list = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # attributions, delta = ig.attribute(inputs, target=labels, return_convergence_delta=True)

        # inputs, labels = inputs.to(device), labels.to(device)
        # heatmap,idx = generate_gradcam(model,inputs)

        attributions = grad_cam(inputs, targets=[ClassifierOutputTarget(labels)])
        # attributions = grad_cam.attribute(inputs)
        # attributions = visualization.rescale(attributions, inputs.size)

        # attributions = torch.nn.functional.interpolate(attributions, size=(inputs.shape[-2], inputs.shape[-1]),
        #                                                mode='bilinear', align_corners=False)

        attributions = np.transpose(attributions, (1, 2, 0))
        # attributions = np.maximum(0, attributions)
        heatmap = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-10)

        # show the case :

        # image = inputs.cpu().detach().numpy()[0]
        # image = image.transpose([1,2,0])
        # image = image* np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        # image = np.clip(image,0,1)
        # plt.imsave('./image.jpg',image)

        # # cv
        # image = image[:,:,::-1]
        # cv2.imwrite('image.jpg',(image*255).astype(np.uint8))
        # heatmap_img = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        # super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, (image*255).astype(np.uint8), 0.5, 0)
        # cv2.imwrite('weighted_image.jpg',super_imposed_img)

        # denormalize
        # plt.imsave('./image.jpg',image)
        # plt.close()
        # plt.figure()
        # plt.imshow(heatmap[:,:,0], cmap='hot', interpolation='nearest')
        # plt.savefig('heatmap.jpg')

        # heatmap = grad_cam(model,inputs,labels)
        # heatmap = generate_gradcam_heatmap(model,labels,inputs)
        explanations_list.append(heatmap)

    np.save('../../data/cifar100/LayerCAM/train_resnet-9-explanations',explanations_list)


if __name__=="__main__":
    explain_ori_model_vis()




