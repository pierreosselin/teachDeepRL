import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Convolutional_Net(nn.Module):
    def __init__(self, hidden_sizes=(150, 250, 150, 100, 50, 32, 4), activation='relu', output_activation=None):
        super().__init__()

        self.conv_layers = []
        self.conv_layers.append(nn.Conv2d(in_channels = 1, out_channels = hidden_sizes[0], kernel_size = 3, padding="same"))
        last_el = hidden_sizes[0]
        for el in hidden_sizes[1:-1]:
            self.conv_layers.append(nn.Conv2d(in_channels = last_el, out_channels = el, kernel_size = 3, padding="same"))
            last_el = el
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(last_el * 17 *17, hidden_sizes[-1])

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x

if __name__== "__main__":
    from teachDRL.gym_flowers.envs.maze_env import *
    model = Convolutional_Net()
    env_config = {}
    env_config['device'] = "cuda"
    env_config['maze_model_path'] = "/home/pierre/Git/teachDeepRL/teachDRL/models/generatornobatch_aldous-pacman_4.pth"
    env = MazeEnv(env_config)
    x = env.maze.cpu()
    print(x.shape)
    print(model(x))