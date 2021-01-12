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
import A02_Networks.FeedForward as FF_Class
import pytorch_lightning as pl
from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser


def calcParam_from_NN(Variant, argsValues):

    if argsValues.Condition == 'Single':

        if Variant == 1:
            print('Single - Variant 1')
            print("Leuchtdichte: " + str(argsValues.L) +
                  " Farbort x: " + str(argsValues.Fx) +
                  " Farbort y: " + str(argsValues.Fy))

            Leuchtdichte = argsValues.L
            Farbort_x = argsValues.Fx
            Farbort_y = argsValues.Fy
            PATH = "A03_Models/FF/Intrapersonal_SingleSubject/Variant_1_Lxy/FF_Variant_1_BatchSize_7_epoch=3999.ckpt"
            model = FF_Class.FeedForward.load_from_checkpoint(PATH)
            model.eval()

            Eingangswerte = np.array(
                [argsValues.L, argsValues.Fx, argsValues.Fy])

            Eingangswerte = torch.from_numpy(Eingangswerte).float()

            output = model(Eingangswerte)

            output_np = output.data.numpy()

            dataset = pd.DataFrame({'f_p': output_np[0],
                                    'f_s': output_np[1],
                                    'P_0': output_np[2],
                                    'tp': output_np[3],
                                    'ts': output_np[4],
                                    'Delta_tp': output_np[5],
                                    'Delta_ts': output_np[6],
                                    'p1': output_np[7],
                                    'p2': output_np[8],
                                    'p3': output_np[9],
                                    'p4': output_np[10],
                                    'p5': output_np[11],
                                    'p6': output_np[12],
                                    'p7': output_np[13],
                                    'p8': output_np[14],
                                    'p9': output_np[15],
                                    'p10': output_np[16]}, index=[0])

            dataset.to_csv('output.csv', index=False)

            # Eingangswerte:
            # 470 nm - Leuchtdichte: 0 Farbort x: 0 Farbort y: 0
            # 530 nm - Leuchtdichte: 0.886363636363642 Farbort x: 0.0509930220075148 Farbort y: 1
            # 610 nm - Leuchtdichte: 0.977272727272716 Farbort x: 0.932045089 Farbort y: 0.932045089
            # 660 nm - Leuchtdichte: 0.545454545 Farbort x: 1 Farbort y: 0.361349796

        if Variant == 2:
            print('Single - Variant 2')
            print("Lcone: " + str(argsValues.Lcone) +
                  " Mcone: " + str(argsValues.Mcone) +
                  " Scone: " + str(argsValues.Scone) +
                  " Mel: " + str(argsValues.Mel))

            Lcone = argsValues.Lcone
            Mcone = argsValues.Mcone
            Scone = argsValues.Scone
            Mel = argsValues.Mel
            PATH = "A03_Models/FF/Intrapersonal_SingleSubject/Variant_2_LMSMEL/FF_Variant_2_BatchSize_7_epoch=2900.ckpt"
            model = FF_Class.FeedForward.load_from_checkpoint(PATH)
            model.eval()

            Eingangswerte = np.array([argsValues.Scone,
                                      argsValues.Mcone,
                                      argsValues.Lcone,
                                      argsValues.Mel])

            Eingangswerte = torch.from_numpy(Eingangswerte).float()

            output = model(Eingangswerte)

            output_np = output.data.numpy()

            dataset = pd.DataFrame({'f_p': output_np[0],
                                    'f_s': output_np[1],
                                    'P_0': output_np[2],
                                    'tp': output_np[3],
                                    'ts': output_np[4],
                                    'Delta_tp': output_np[5],
                                    'Delta_ts': output_np[6],
                                    'p1': output_np[7],
                                    'p2': output_np[8],
                                    'p3': output_np[9],
                                    'p4': output_np[10],
                                    'p5': output_np[11],
                                    'p6': output_np[12],
                                    'p7': output_np[13],
                                    'p8': output_np[14],
                                    'p9': output_np[15],
                                    'p10': output_np[16]}, index=[0])

            dataset.to_csv('output.csv', index=False)

        if Variant == 3:
            print('Single - Variant 3')
            print("Leuchtdichte: " + str(argsValues.L) +
                  " Farbort x: " + str(argsValues.Fx) +
                  " Farbort y: " + str(argsValues.Fy) +
                  " Mel: " + str(argsValues.Mel))

            Leuchtdichte = argsValues.L
            Farbort_x = argsValues.Fx
            Farbort_y = argsValues.Fy
            Mel = argsValues.Mel

            PATH = "A03_Models/FF/Intrapersonal_SingleSubject/Variant_3_LxyMel/FF_Variant_3_BatchSize_7_epoch=3999.ckpt"
            model = FF_Class.FeedForward.load_from_checkpoint(PATH)
            model.eval()

            Eingangswerte = np.array([argsValues.L,
                                      argsValues.Fx,
                                      argsValues.Fy,
                                      argsValues.Mel])

            Eingangswerte = torch.from_numpy(Eingangswerte).float()

            output = model(Eingangswerte)

            output_np = output.data.numpy()

            dataset = pd.DataFrame({'f_p': output_np[0],
                                    'f_s': output_np[1],
                                    'P_0': output_np[2],
                                    'tp': output_np[3],
                                    'ts': output_np[4],
                                    'Delta_tp': output_np[5],
                                    'Delta_ts': output_np[6],
                                    'p1': output_np[7],
                                    'p2': output_np[8],
                                    'p3': output_np[9],
                                    'p4': output_np[10],
                                    'p5': output_np[11],
                                    'p6': output_np[12],
                                    'p7': output_np[13],
                                    'p8': output_np[14],
                                    'p9': output_np[15],
                                    'p10': output_np[16]}, index=[0])

            dataset.to_csv('output.csv', index=False)

    if argsValues.Condition == 'Multi':
        if Variant == 1:
            print('Many - Variant 1')
            print("Leuchtdichte: " + str(argsValues.L) +
                  " Farbort x: " + str(argsValues.Fx) +
                  " Farbort y: " + str(argsValues.Fy))

            Leuchtdichte = argsValues.L
            Farbort_x = argsValues.Fx
            Farbort_y = argsValues.Fy
            PATH = "A03_Models/FF/Interpersonal_ManySubject/Variant_1_Lxy/FF_Variant_1_BatchSize_7_epoch=800.ckpt"
            model = FF_Class.FeedForward.load_from_checkpoint(PATH)
            model.eval()

            Eingangswerte = np.array(
                [argsValues.L, argsValues.Fx, argsValues.Fy])

            Eingangswerte = torch.from_numpy(Eingangswerte).float()

            output = model(Eingangswerte)

            output_np = output.data.numpy()

            dataset = pd.DataFrame({'f_p': output_np[0],
                                    'f_s': output_np[1],
                                    'P_0': output_np[2],
                                    'tp': output_np[3],
                                    'ts': output_np[4],
                                    'Delta_tp': output_np[5],
                                    'Delta_ts': output_np[6],
                                    'p1': output_np[7],
                                    'p2': output_np[8],
                                    'p3': output_np[9],
                                    'p4': output_np[10],
                                    'p5': output_np[11],
                                    'p6': output_np[12],
                                    'p7': output_np[13],
                                    'p8': output_np[14],
                                    'p9': output_np[15],
                                    'p10': output_np[16]}, index=[0])

            dataset.to_csv('output.csv', index=False)

            # Eingangswerte:
            # 470 nm - Leuchtdichte: 0 Farbort x: 0 Farbort y: 0
            # 530 nm - Leuchtdichte: 0.886363636363642 Farbort x: 0.0509930220075148 Farbort y: 1
            # 610 nm - Leuchtdichte: 0.977272727272716 Farbort x: 0.932045089 Farbort y: 0.932045089
            # 660 nm - Leuchtdichte: 0.545454545 Farbort x: 1 Farbort y: 0.361349796

        if Variant == 2:
            print('Many - Variant 2')
            print("Lcone: " + str(argsValues.Lcone) +
                  " Mcone: " + str(argsValues.Mcone) +
                  " Scone: " + str(argsValues.Scone) +
                  " Mel: " + str(argsValues.Mel))

            Lcone = argsValues.Lcone
            Mcone = argsValues.Mcone
            Scone = argsValues.Scone
            Mel = argsValues.Mel
            PATH = "A03_Models/FF/Interpersonal_ManySubject/Variant_2_LMSMEL/FF_Variant_2_BatchSize_7_epoch=3800.ckpt"
            model = FF_Class.FeedForward.load_from_checkpoint(PATH)
            model.eval()

            Eingangswerte = np.array([argsValues.Scone,
                                      argsValues.Mcone,
                                      argsValues.Lcone,
                                      argsValues.Mel])

            Eingangswerte = torch.from_numpy(Eingangswerte).float()

            output = model(Eingangswerte)

            output_np = output.data.numpy()

            dataset = pd.DataFrame({'f_p': output_np[0],
                                    'f_s': output_np[1],
                                    'P_0': output_np[2],
                                    'tp': output_np[3],
                                    'ts': output_np[4],
                                    'Delta_tp': output_np[5],
                                    'Delta_ts': output_np[6],
                                    'p1': output_np[7],
                                    'p2': output_np[8],
                                    'p3': output_np[9],
                                    'p4': output_np[10],
                                    'p5': output_np[11],
                                    'p6': output_np[12],
                                    'p7': output_np[13],
                                    'p8': output_np[14],
                                    'p9': output_np[15],
                                    'p10': output_np[16]}, index=[0])

            dataset.to_csv('output.csv', index=False)

        if Variant == 3:
            print('Many - Variant 3')
            print("Leuchtdichte: " + str(argsValues.L) +
                  " Farbort x: " + str(argsValues.Fx) +
                  " Farbort y: " + str(argsValues.Fy) +
                  " Mel: " + str(argsValues.Mel))

            Leuchtdichte = argsValues.L
            Farbort_x = argsValues.Fx
            Farbort_y = argsValues.Fy
            Mel = argsValues.Mel

            PATH = "A03_Models/FF/Interpersonal_ManySubject/Variant_3_LxyMel/FF_Variant_3_BatchSize_7_epoch=3800.ckpt"
            model = FF_Class.FeedForward.load_from_checkpoint(PATH)
            model.eval()

            Eingangswerte = np.array([argsValues.L,
                                      argsValues.Fx,
                                      argsValues.Fy,
                                      argsValues.Mel])

            Eingangswerte = torch.from_numpy(Eingangswerte).float()

            output = model(Eingangswerte)

            output_np = output.data.numpy()

            dataset = pd.DataFrame({'f_p': output_np[0],
                                    'f_s': output_np[1],
                                    'P_0': output_np[2],
                                    'tp': output_np[3],
                                    'ts': output_np[4],
                                    'Delta_tp': output_np[5],
                                    'Delta_ts': output_np[6],
                                    'p1': output_np[7],
                                    'p2': output_np[8],
                                    'p3': output_np[9],
                                    'p4': output_np[10],
                                    'p5': output_np[11],
                                    'p6': output_np[12],
                                    'p7': output_np[13],
                                    'p8': output_np[14],
                                    'p9': output_np[15],
                                    'p10': output_np[16]}, index=[0])

            dataset.to_csv('output.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--Condition', type=str, default='Single')
    parser.add_argument('--Variant', type=int, default=1)
    parser.add_argument('--L', type=float, default=0)
    parser.add_argument('--Fx', type=float, default=0)
    parser.add_argument('--Fy', type=float, default=0)
    parser.add_argument('--Lcone', type=float, default=0)
    parser.add_argument('--Mcone', type=float, default=0)
    parser.add_argument('--Scone', type=float, default=0)
    parser.add_argument('--Mel', type=float, default=0)

    args = parser.parse_args()
    calcParam_from_NN(Variant=args.Variant, argsValues=args)
    print("Values calculated and exported to csv")

    # Zum ausführen der verschiedenen Varianten müssen folgende Befehle eingegeben werden
    # Variante 1: python CalcParam.py --Condition Single --Variant 1 --L 0 --Fx 0 --Fy 0
    # Variante 2: python CalcParam.py --Condition Single --Variant 2 --Lcone 0 --Mcone 0 --Scone 0 --Mel 0
    # Variante 3: python CalcParam.py --Condition Single --Variant 3 --L 0 --Fx 0 --Fy 0 --Mel 0

    # Variante 1: python CalcParam.py --Condition Multi --Variant 1 --L 0 --Fx 0 --Fy 0
    # Variante 2: python CalcParam.py --Condition Multi --Variant 2 --Lcone 0 --Mcone 0 --Scone 0 --Mel 0
    # Variante 3: python CalcParam.py --Condition Multi --Variant 1 --L 0 --Fx 0 --Fy 0 --Mel 0
