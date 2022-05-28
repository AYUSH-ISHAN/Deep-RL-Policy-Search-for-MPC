
from pickletools import optimize
from turtle import forward
import torch.nn as nn
import numpy as np
import glob
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
from torch.nn.modules import linear
from torch.optim import Adam
from policy.high_policy import GaussianPolicy
from dataset_maker import Dataset


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_units):
        super(PolicyNet, self).__init__()
        self.input = nn.Linear(obs_dim, hidden_units[0])
        self.hid = nn.Linear(hidden_units[0], hidden_units[1])
        self.out = nn.Linear(hidden_units[1], act_dim)

    def forward(self, states):
        x = F.relu(self.input(states))
        x = F.relu(self.hid(x))
        out = self.out(x)

        return out

class Actor(PolicyNet):

    def __init__(self, obs_dim, act_dim, \
        learning_rate=3e-4, hidden_units=[32, 32],  \
        activation='relu', trainable=True, actor_name="actor"):
        super(Actor, self).__init__(obs_dim, act_dim, hidden_units)
        # #
        # obs = Input(shape=(obs_dim, ))
        self.model = PolicyNet(obs_dim, act_dim, hidden_units)
        # act = self.model.forward(obs)
        # #
        #self._act_net = Model(inputs=obs, outputs=act)
        self._optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def call(self, inputs):
        return self.model.forward(inputs)

    def save_weights(self, save_dir, iter):
        save_dir = save_dir + "/act_net"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.weights_path = save_dir + "/weights_{0}.h5".format(iter)
        # torch.save(self.model, self.weights_path)
        torch.save(self.model.state_dict(), self.weights_path)
        # self._act_net.save_weights(weights_path)

    def load_weights(self, file_path):
        # self._act_net.load_weights(file_path)
        # self.model = PolicyNet(*args, **kwargs)
        print("weights path : ", file_path)
        self.model.load_state_dict(torch.load(file_path))
        # self.model.eval()
        # self.model=torch.load(self.weights_path)
        # return self.model

    #@tf.function
    def train_batch(self, obs_batch, act_batch):
        # with tf.GradientTape(persistent=True) as tape:
        error = nn.MSELoss()
        losses = 0
        for (obs, act) in zip(obs_batch, act_batch):
             act_pred = self.call(obs)
        #     # mean squared error
             mse_loss = error(act_pred, act)
             losses+=mse_loss
             mse_loss.backward()
             self._optimizer.step()
             self._optimizer.zero_grad()

        # #   
        # act_grad = tape.gradient(mse_loss, self._act_net.trainable_variables)
        # self._optimizer.apply_gradients( zip(act_grad, self._act_net.trainable_variables))
        
        return losses


def train(env, logger, data_dir, hidden_units, learning_rate, \
     activation, train_epoch, batch_size):

    obs_dim = 16 #env.observation_space.shape[0]
    act_dim = 4 #env.action_space.shape[0]
    data_dir = "./Dataset/deep_highmpc-05-26-15-47-14/dataset/"
    print(data_dir)
    print(glob.glob(os.path.join(data_dir, "*.npz")))
    # dataset = Dataset(obs_dim, act_dim)
    # dataset.load(data_dir)
    train_obs, train_act = None, None
    for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "*.npz"))):
        size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
        print(size)
        np_file = np.load(data_file)
        #
        obs_array = np_file['obs'][:size, :]
        act_array = np_file['act'][:size, :]
        if i == 0:
            train_obs = obs_array
            train_act = act_array    
        else:
            train_obs = np.append(train_obs, obs_array, axis=0)
            train_act = np.append(train_act, act_array, axis=0)
    #
    tensor_x = torch.Tensor(train_obs) # transform to torch tensor
    tensor_y = torch.Tensor(train_act)
    
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset) # create your dataloader
    
    # train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_obs, train_act))
    #
    actor = Actor(obs_dim, act_dim)
    #
    # trainer.train()
    SHUFFLE_BUFFER_SIZE = 100
    for epoch in range(train_epoch):
        #
        train_loss = []
        # dataset = train_tf_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        for obs, act in my_dataloader:#dataset.batch(batch_size, drop_remainder=False):
            loss = actor.train_batch(obs, act)
            train_loss.append(loss.detach().numpy())

        if (epoch) % 100 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch, np.mean(train_loss)))
            actor.save_weights(data_dir, iter=epoch)


    
