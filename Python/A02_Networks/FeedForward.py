# Description:
# Demo code from the article:
# Deep learning based pupil model predicts time and wavelength dependent light responses
# Technical University of Darmstadt, Laboratory of Lighting Technology
# Published in Scientific Reports
# Link: www.nature.com/articles/s41598-020-79908-5
# GitHub Link: https://github.com/BZandi/DL-PupilModel


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import torch.utils.data
import time
import pytorch_lightning as pl
from A01_Functions.ReadData import *
from argparse import Namespace

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 600)


class FeedForward(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.Variant = hparams.Variant
        self.BatchSize = hparams.BatchSize
        self.learningRate = hparams.learningRate
        self.InputSize = hparams.InputSize

        self.input_Layer = nn.Linear(self.InputSize, hparams.layer_1_dim)
        self.hidden_layer_1 = nn.Linear(hparams.layer_1_dim, 4 * hparams.layer_2_dim)
        self.hidden_layer_2 = nn.Linear(4 * hparams.layer_2_dim, hparams.layer_2_dim)
        self.output_Layer = nn.Linear(hparams.layer_2_dim, 17)

    def forward(self, x):
        x = self.input_Layer(x)
        x = self.hidden_layer_1(F.relu(x))
        x = self.hidden_layer_2(F.relu(x))
        x = self.output_Layer(x)

        return x

    def prepare_data(self):
        [self.TrainInputMatrix, self.TrainTargetMatrix] = Func_readDataIn(self.Variant)
        self.trainData = torch.utils.data.TensorDataset(self.TrainInputMatrix, self.TrainTargetMatrix)

    def train_dataloader(self):

        return torch.utils.data.DataLoader(dataset=self.trainData, batch_size=self.BatchSize,
                                           shuffle=True, drop_last=True, num_workers=4)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learningRate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        criterion = nn.MSELoss()
        criterion_2 = nn.L1Loss()
        loss = criterion(output, target)
        #MAE = abs(output - target).mean()
        MAE = criterion_2(output, target).mean()
        SD = abs(output - target).std()
        # add logging
        # logs = {'loss': loss}
        # return {'loss': loss, 'log': logs}
        return {'loss': loss, 'MAE': MAE, 'SD': SD}


class MyCallback(pl.Callback):
    # TODO:
    # Werte in einer CSV speichern zum späteren zugreifen
    # Wshrscheinlich kann man im training step auf das Objekt zugreifen
    def __init__(self, PlotName, DataName, ModelPath):
        self.PlotName = PlotName
        self.DataName = DataName
        self.ModelPath = ModelPath

    def on_init_start(self, trainer):
        self.hold_epoch = []
        self.hold_MAE_Train = []
        self.hold_MSE_Train = []
        self.hold_SD_Train = []

    def on_batch_start(self, trainer, pl_module):
        pass

    def on_batch_end(self, trainer, pl_module):
        # Wenn in dicct was drin ist
        if bool(trainer.callback_metrics) == True:
            self.Mean_hold_MSE.append(trainer.callback_metrics.get("loss"))
            self.Mean_hold_MAE.append(trainer.callback_metrics.get("MAE"))
            self.Mean_hold_SD.append(trainer.callback_metrics.get("SD"))

    def on_epoch_start(self, trainer, pl_module):
        self.start = time.time()
        self.Mean_hold_MSE = []
        self.Mean_hold_MAE = []
        self.Mean_hold_SD = []
        pass

    def on_epoch_end(self, trainer, pl_module):
        end = time.time()
        if len(self.Mean_hold_MSE) > 0:
            print('Train Epoch [{}/{}]: Loss (MSE): {:.9f} Loss (MAE): {:.9f} SD {:.8f} - Time:{:.3f} Seconds'.
                  format(trainer.current_epoch,
                         trainer.max_epochs,
                         np.mean(self.Mean_hold_MSE),
                         np.mean(self.Mean_hold_MAE),
                         np.mean(self.Mean_hold_SD),
                         end - self.start))

            self.hold_epoch.append(trainer.current_epoch)
            self.hold_MSE_Train.append(np.mean(self.Mean_hold_MSE))
            self.hold_MAE_Train.append(np.mean(self.Mean_hold_MAE))
            self.hold_SD_Train.append(np.mean(self.Mean_hold_SD))

            fig, ax = plt.subplots()
            plt_MAE_Mean = ax.plot(self.hold_epoch, self.hold_MAE_Train, 'b', label='Mean absolute error')
            plt_MSE = ax.plot(self.hold_epoch, self.hold_MSE_Train, 'k', label='Mean squared error')
            # plt_SD = ax.fill_between(self.hold_epoch,
            #                         np.array(self.hold_MAE_Train) + np.array(self.hold_SD_Train),
            #                         np.array(self.hold_MAE_Train) - np.array(self.hold_SD_Train),
            #                         color='yellow', alpha=0.5, label='Standard deviation')
            plt.yscale('log', nonposy='clip')
            plt.legend(loc='upper center')
            fig.savefig('A04_Results_Training/01_Plots/' + self.PlotName + '.png')
            plt.close()
            start = time.time()

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.ModelPath + 'epoch=' + str(trainer.current_epoch) + '.ckpt')
        dataset = pd.DataFrame({'Epoch': self.hold_epoch, 'MSE': self.hold_MSE_Train,
                                'MAE': self.hold_MAE_Train, 'SD': self.hold_SD_Train})

        dataset.to_csv('A04_Results_Training/02_Data/' + self.DataName + '.csv', index=False)
