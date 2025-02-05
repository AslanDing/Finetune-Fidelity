import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Define CNN architecture
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

# init model
model = CNNClassifier()
# load fintuned model
# model.load_state_dict(torch.load("./path_to_state_dict.pth"))


# init a picture 
input = np.random.uniform(size = [1,3,32,32])/0.5 - 1  # assume the input is already normalized
label = 0 
# explanation
explanation_score = np.random.uniform(size = [32,32])  # explanation for each pxiel



# the paper used the label-based fidelity, it is easy to extend to the probability-based version
# the original out prediction
output = model(torch.from_numpy(input).float())
output = output.argmax().item() 
if math.fabs(output - label) < 0.5:
    init_prediction = 1
else:
    init_prediction = 0



# define top_k 
top_k_ratio = s = 0.1 

# get top_k mask
scores = explanation_score.reshape(-1)
num = int(top_k_ratio * scores.shape[0])
select = np.arange(scores.shape[0])
idx_select = np.random.choice(select, num, replace=False)
mask = np.zeros_like(scores)
mask[idx_select] = 1


# define alpha, beta
all_pxiels = input.shape[-2] * input.shape[-1]
alpha = 0.5
beta = 0.1   # during finetune, the random drop ratio is beta
num_samples = 50






# cacluate the fid+
masks = torch.from_numpy( 1 - mask ).view(1, 1, input.shape[-2], input.shape[-1])

ratio = alpha
if s*alpha> beta * all_pxiels:
    ratio = beta * all_pxiels/ s

union_masks = torch.rand([num_samples, masks.shape[-3], masks.shape[-2], masks.shape[-1]])
union_masks = torch.where(union_masks > ratio, torch.ones_like(union_masks),
                            torch.zeros_like(union_masks)) 
new_masks = masks + (1 - masks) * union_masks

new_inputs = torch.tile(torch.from_numpy(input).float(), [num_samples, 1, 1, 1])
new_inputs = new_inputs * new_masks

new_outputs = model(new_inputs.float())
new_outputs = new_outputs.argmax(dim=-1)
 
results = new_outputs.float() - label
results = torch.where(torch.abs(results) < 0.5, torch.ones_like(results), torch.zeros_like(results))

ffid_plus = init_prediction - results.mean().item()


# cacluate the fid-
masks = torch.from_numpy( mask ).view(1, 1, input.shape[-2], input.shape[-1])

ratio = alpha
if s*alpha> beta * all_pxiels:
    ratio = beta * all_pxiels/ s

union_masks = torch.rand([num_samples, masks.shape[-3], masks.shape[-2], masks.shape[-1]])
union_masks = torch.where(union_masks > ratio, torch.ones_like(union_masks),
                            torch.zeros_like(union_masks)) 
new_masks = masks + (1 - masks) * union_masks

new_inputs = torch.tile(torch.from_numpy(input).float(), [num_samples, 1, 1, 1])
new_inputs = new_inputs * new_masks

new_outputs = model(new_inputs.float())
new_outputs = new_outputs.argmax(dim=-1)
 
results = new_outputs.float() - label
results = torch.where(torch.abs(results) < 0.5, torch.ones_like(results), torch.zeros_like(results))

ffid_minus = init_prediction - results.mean().item()


print(ffid_plus,ffid_minus)


