import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class PPO_Net(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)