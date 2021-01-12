# Description:
# Demo code from the article:
# Deep learning based pupil model predicts time and wavelength dependent light responses
# Technical University of Darmstadt, Laboratory of Lighting Technology
# Published in Scientific Reports
# Link: www.nature.com/articles/s41598-020-79908-5
# GitHub Link: https://github.com/BZandi/DL-PupilModel

import numpy as np
import pandas as pd
import sys
import torch

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 600)
np.set_printoptions(linewidth=200, edgeitems=4)
torch.set_printoptions(linewidth=200, edgeitems=4)


def Func_readDataIn(Variant):
    DataDataFrame = pd.read_csv('A00_Data/TrainData_Many_Subject.csv')
    print('Data loaded from csv')

    [InputtLabels, TargetLabels] = shapeTrainData(Variant)

    print('Features: ' + str(InputtLabels))
    print('Target: ' + str(TargetLabels))

    InputValues = DataDataFrame[InputtLabels]
    TargetValues = DataDataFrame[TargetLabels]

    InputValues = InputValues.values
    TargetValues = TargetValues.values

    InputValues = torch.from_numpy(InputValues).float()
    TargetValues = torch.from_numpy(TargetValues).float()

    return [InputValues, TargetValues]


def shapeTrainData(Variant):
    # Input: Leuchtdichte, Farbort_x, Farbort_y
    # Output: f_p, f_s, P_0, tp, ts, Delta_tp, Delta_ts, p1
    #         p2, p3, p4, p5, p6, p7, p8, p9, p10
    if Variant == 1:
        InputtLabels = ['Leuchtdichte', 'Farbort_x', 'Farbort_y']
        TargetLabels = ['f_p', 'f_s', 'P_0', 'tp', 'ts', 'Delta_tp', 'Delta_ts',
                        'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']

    if Variant == 2:
        InputtLabels = ['S_Signal', 'M_Signal', 'L_Signal', 'Melanopsin_Signal']
        TargetLabels = ['f_p', 'f_s', 'P_0', 'tp', 'ts', 'Delta_tp', 'Delta_ts',
                        'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']

    if Variant == 3:
        InputtLabels = ['Leuchtdichte', 'Farbort_x', 'Farbort_y', 'Melanopsin_Signal']
        TargetLabels = ['f_p', 'f_s', 'P_0', 'tp', 'ts', 'Delta_tp', 'Delta_ts',
                        'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']

    return InputtLabels, TargetLabels
