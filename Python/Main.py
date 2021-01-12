# Description:
# Demo code from the article:
# Deep learning based pupil model predicts time and wavelength dependent light responses
# Technical University of Darmstadt, Laboratory of Lighting Technology
# Published in Scientific Reports
# Link: www.nature.com/articles/s41598-020-79908-5
# GitHub Link: https://github.com/BZandi/DL-PupilModel


import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from A01_Functions.ReadData import*
import A02_Networks.FeedForward as FF_Class
import pytorch_lightning as pl
from argparse import Namespace
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 600)
np.set_printoptions(linewidth=200, edgeitems=4)
torch.set_printoptions(linewidth=200, edgeitems=4)


def Func_runTraining(Epoch, numHiddenNeuron_1, numHiddenNeuron_2, Variant,
                     BatchSize, learningRate):

    if Variant == 1:
        InputSize = 3
    if Variant == 2:
        InputSize = 4
    if Variant == 3:
        InputSize = 4

    hparams = Namespace(layer_1_dim=numHiddenNeuron_1,
                        layer_2_dim=numHiddenNeuron_2,
                        Variant=Variant,
                        BatchSize=BatchSize,
                        learningRate=learningRate,
                        InputSize=InputSize)

    # **{"layer_1_dim": numHiddenNeuron_1,
    #        "layer_2_dim": numHiddenNeuron_2,
    #        "Variant": Variant,
    #        "BatchSize": BatchSize,
    #        "learningRate": learningRate})

    Model = FF_Class.FeedForward(hparams)

    PlotNameStr = 'FF_Variant_' + str(Variant) + '_BatchSize_' + str(BatchSize) + '_Epoch_' + str(Epoch) +\
                  '_Hidden_' + str(numHiddenNeuron_1) + '_' + str(numHiddenNeuron_2)

    CheckpointNameStr = 'FF_Variant_' + str(Variant) + '_BatchSize_' + str(BatchSize) + '_'

    ModelPath = 'A03_Models/FF/' + CheckpointNameStr

    checkpoint_callback = ModelCheckpoint(filepath='A03_Models/FF/' + CheckpointNameStr + '{epoch}',
                                          save_top_k=-1,
                                          period=100)

    trainer = pl.Trainer(progress_bar_refresh_rate=0, logger=False,
                         checkpoint_callback=checkpoint_callback,  # oder False
                         callbacks=[FF_Class.MyCallback(PlotName=PlotNameStr, DataName=PlotNameStr,
                                                        ModelPath=ModelPath)],
                         max_epochs=Epoch, reload_dataloaders_every_epoch=False)
    trainer.fit(Model)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--Variant', type=int, default=1)
    args = parser.parse_args()

    Epoch = 4000
    numHiddenNeuron_1 = 40
    numHiddenNeuron_2 = 80
    BatchSize = 7
    learningRate = 0.001

    print("Train values - " + "Variant: " + str(args.Variant) +
          "Epoch:" + str(Epoch) +
          " numHiddenNeuron_1: " + str(numHiddenNeuron_1) +
          " numHiddenNeuron_2: " + str(numHiddenNeuron_2) +
          " BatchSize: " + str(BatchSize) +
          " learningRate: " + str(learningRate))

    Func_runTraining(Epoch=Epoch, numHiddenNeuron_1=numHiddenNeuron_1, numHiddenNeuron_2=numHiddenNeuron_2,
                     Variant=args.Variant, BatchSize=BatchSize, learningRate=learningRate)
