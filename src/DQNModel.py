import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class Network3D(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier=True):
        super(Network3D, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(in_channels=frame_history,out_channels=32,kernel_size=(5, 5, 5),padding=1).to(self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)

        self.conv1 = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=(5, 5, 5),padding=1).to(self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)

        self.conv2 = nn.Conv3d(in_channels=32,out_channels=64,kernel_size=(4, 4, 4),padding=1).to(self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=(3, 3, 3),padding=0).to(self.device)

        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=512, out_features=256).to(
                self.device) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=256, out_features=128).to(
                self.device) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=128, out_features=number_actions).to(
                self.device) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input = input.to(self.device) / 255.0
        output = []
        for i in range(self.agents):
            # Shared layers
            x = input[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            x = x.view(-1, 512)
            # Individual layers
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            output.append(x)
        output = torch.stack(output, dim=1)
        return output.cpu()

########################################################################################

class CommNet(nn.Module):

    def __init__(self, agents, landmarks, frame_history, number_actions=4, xavier=True, attention=False):
        super(CommNet, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if number_actions == 6: #3D
            self.conv0 = nn.Conv3d(in_channels=frame_history, out_channels=32, kernel_size=(5, 5, 5),padding=1).to(self.device)
            self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
            self.conv1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5, 5, 5), padding=1).to(self.device)
            self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
            self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 4), padding=1).to(self.device)
            self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
            self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=0).to(self.device)
        else: #2D
            # (64,1,4,61,61) conv out: (Size+2*padd-dil*(k-1)-1)/s)+1, round down
            self.conv0 = nn.Conv2d(
                in_channels=frame_history,
                out_channels=32,
                kernel_size=(5, 5),
                padding=1).to(self.device)  #(32,61-2,59)
            self.maxpool0 = nn.MaxPool2d(
                kernel_size=(2, 2)).to(self.device) #(32,(59-2)/2 +1,29)

            self.conv1 = nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(5, 5),
                padding=1).to(self.device) #(32,27,27)
            self.maxpool1 = nn.MaxPool2d(
                kernel_size=(2, 2)).to(self.device) #(32,13,13)

            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4), padding=1).to(self.device) #(64,13+2-4+1,12)
            self.maxpool2 = nn.MaxPool2d(
                kernel_size=(2, 2)).to(self.device)  # (64,6,6)

            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=0).to(self.device)  # (64,4,4)

        # CONV output: (64,4,4)
        self.prelu0 = nn.PReLU().to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(
                in_features= 64*4*4 * 2,
                out_features= 256).to(
                self.device) for _ in range(
                self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256 * 2,
                out_features=128).to(
                self.device) for _ in range(
                self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128 * 2,
                out_features=number_actions).to(
                self.device) for _ in range(
                self.agents)])

        self.attention = attention
        if self.attention:
                self.comm_att1 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att2 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att3 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        # Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device) / 255.0

        # Shared layers
        input2 = []
        for i in range(self.agents):
            x = input1[:, i]    # (64,1,4,61,61)
            x = self.conv0(x)
            # x = self.conv0(x.float())   # (64,1,32,59,59)
            x = self.prelu0(x)
            x = self.maxpool0(x)    #(64,1,32,30,30)
            x = self.conv1(x)       #(32,27,27)
            x = self.prelu1(x)
            x = self.maxpool1(x)    #(32,13,13)
            x = self.conv2(x)       #(64,12,12)
            x = self.prelu2(x)
            x = self.maxpool2(x)    #(64,6,6)
            x = self.conv3(x)       #(64,4,4)
            x = self.prelu3(x)
            x = x.view(-1, 64*4*4)     # (64,2,512)
            input2.append(x)
        # output (64,2,512)
        input2 = torch.stack(input2, dim=1)  # batch-size, agents,
         
        # Communication layers
        if self.attention:
            comm = torch.cat([torch.sum((input2.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att1[i])), dim=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input2, dim=1)  # (64,1,512)
            # comm = torch.mean(input2, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))  # (agents, 64,1,512)
        
        input3 = []
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](torch.cat((x, comm[i]), dim=-1))
            # x = self.fc1[i](torch.cat((x, comm[i]), axis=-1))
            input3.append(self.prelu4[i](x))
        input3 = torch.stack(input3, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input3.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att2[i])), dim=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input3, dim=1)
            # comm = torch.mean(input3, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input4 = []
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](torch.cat((x, comm[i]), dim=-1))
            # x = self.fc2[i](torch.cat((x, comm[i]), axis=-1))
            input4.append(self.prelu5[i](x))
        input4 = torch.stack(input4, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input4.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att3[i])), dim=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input4, dim=1)
            # comm = torch.mean(input4, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        
        output = []
        for i in range(self.agents):
            x = input4[:, i]
            x = self.fc3[i](torch.cat((x, comm[i]), dim=-1))
            # x = self.fc3[i](torch.cat((x, comm[i]), axis=-1))
            output.append(x)
        output = torch.stack(output, dim=1)

        return output.cpu()

class DQN:
    # The class initialisation function.
    def __init__(self, agents, frame_history, logger, number_actions=4, merge_layers=False,
            type="Network3d", collective_rewards=False, attention=False,
            lr=1e-3, scheduler_gamma=0.9, scheduler_step_size=100, ids=None):

        if merge_layers:
            self.agents = len(np.unique(np.asarray(ids)))
        else:
            self.agents = agents

        self.number_actions = number_actions
        self.frame_history = frame_history
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log(f"Using {self.device}")
        self.loss_func = LossFunction()
        # Create a Q-network, which predicts the q-value for a particular state
        if type == "Network3d":
            self.q_network = Network3D(
                agents,
                frame_history,
                number_actions).to(
                self.device)
            self.target_network = Network3D(
                agents, frame_history, number_actions).to(
                self.device)
        elif type == "CommNet":
            self.q_network = CommNet(
                agents,
                frame_history,
                number_actions,
                attention=attention).to(
                self.device)
            self.target_network = CommNet(
                agents,
                frame_history,
                number_actions,
                attention=attention).to(
                self.device)
        if collective_rewards == "attention":
            self.q_network.rew_att = nn.Parameter(torch.randn(agents, agents))
            self.target_network.rew_att = nn.Parameter(torch.randn(agents, agents))
        self.copy_to_target_network()
        # Freezes target network
        self.target_network.train(False)
        for p in self.target_network.parameters():
            p.requires_grad = False
        # Define the optimiser which is used when updating the Q-network. The
        # learning rate determines how big each gradient step is during
        # backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser, step_size=scheduler_step_size, gamma=scheduler_gamma)
        # print(self.scheduler.state_dict().keys())
        self.collective_rewards = collective_rewards

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, name="dqn.pt", forced=True):
        self.logger.save_model(self.q_network.state_dict(), name, forced)

    # Function that is called whenever we want to train the Q-network. Each
    # call to this function takes in a transition tuple containing the data we
    # use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the
        # loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        '''
        Transitions are tuple of shape
        (states, actions, rewards, next_states, isOver)
        '''
        curr_state = torch.tensor(transitions[0])  # states
        next_state = torch.tensor(transitions[3])  # next states
        terminal = torch.tensor(transitions[4]).type(torch.float)
        # terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(
            torch.tensor(
                transitions[2], dtype=torch.float32), -1, 1)  # rewards

        # Collective rewards here refers to adding the (potentially weighted) average reward of all agents
        if self.collective_rewards == "mean":
            rewards += torch.mean(rewards, dim=1).unsqueeze(1).repeat(1, rewards.shape[1])
        elif self.collective_rewards == "attention":
            rewards = rewards + torch.matmul(rewards, nn.Softmax(dim=0)(self.q_network.rew_att))

        y = self.target_network.forward(next_state)
        # dim (batch_size, agents, number_actions)
        y = y.view(-1, self.agents, self.number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]

        # dim (batch_size, agents, number_actions)
        network_prediction = self.q_network.forward(curr_state).view(
            -1, self.agents, self.number_actions)
        isNotOver = (torch.ones(*terminal.shape) - terminal)

        # Bellman equation, discount_factor=gamma
        batch_labels_tensor = rewards + isNotOver * \
            (discount_factor * max_target_net.detach())

        actions = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = torch.gather(network_prediction, -1, actions).squeeze()

        return self.loss_func.forward(network_pred=network_prediction,
                                    bellman=batch_labels_tensor, pred=y_pred)

class LossFunction(nn.Module):
    def __init__(self, beta=0.01):
        super(LossFunction, self).__init__()
        self.beta = beta

    def forward(self, network_pred, bellman, pred):
        p = Categorical(logits=network_pred)

        dist_loss = torch.nn.SmoothL1Loss()(bellman.flatten(), pred.flatten())
        entropy_loss = -p.entropy().view(-1).sum()
        # print("LOSS FUNCTION VALUES:")
        # print(dist_loss)
        # print(self.beta * entropy_loss)
        return dist_loss + (self.beta * entropy_loss)

