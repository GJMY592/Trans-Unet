# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:30:09 2021

@author: GJMY
"""
import os
import logging
from TSUnet import TSU
import torch
import torch.nn as nn
import GPUtil
import argparse
from dataset_synapse import Synapse_dataset
# from torch.nn.modules.loss import CrossEntropyLoss
from utils import BCELoss2d,SoftDiceLoss,dice_coeff
import pandas as pd

data_size=512
parser = argparse.ArgumentParser(description='Trans-Unet ')
parser.add_argument('--cuda', action='store_true', help='Choose device to use cpu cuda:0')
parser.add_argument('--batch_size', action='store', type=int,
                        default=1, help='number of data in a batch')
parser.add_argument('--lr', action='store', type=float,
                        default=0.1, help='initial learning rate')
parser.add_argument('--epochs', action='store', type=int,
                        default=50, help='train rounds over training set')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')




if __name__ == "__main__":
    opts = parser.parse_args() # Namespace object

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    GPUtil.showUtilization()        
    dataset_train = Synapse_dataset('./','Pancreas-CT','train')
    dataset_test = Synapse_dataset('./','Pancreas-CT','test')
    base_lr = opts.lr
    snapshot_path='./ts_model'
    
    
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=0)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=0)


    model = TSU()
    model.to(device)
    bce_loss = BCELoss2d()
    dice_loss = SoftDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=opts.lr,betas=(0.9,0.99))
    
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    iter_num=0
    max_iterations=opts.epochs*len(data_loader_train.dataset)
    
    model.train()
    for epoch in range(opts.epochs):
        train_batch_num = 0
        train_loss = 0.0
        counts=0
        print("train_round.............%d"%epoch)
        
        for seq, label in data_loader_train:
            seq = seq.to(device)
            label = label.to(device)
            seq = seq.unsqueeze(1)
            optimizer.zero_grad()

            pred = model(seq).squeeze(1)
            pred = (pred-pred.min())/(pred.max()-pred.min())
            
            
            # accuracy
            loss_bce = bce_loss(pred, label)
            loss_dice = dice_loss(pred, label)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            loss.backward()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            opts.lr=lr_
            optimizer.step()
            train_batch_num += 1
            train_loss += loss.item()
            
            
            
            count_cur=dice_coeff(pred,label)
            
            print("Lrate: %.4f"%lr_,"DICE: %.4f"%count_cur)
            counts+=count_cur
            iter_num=iter_num+1
            GPUtil.showUtilization()
        avg_acc = counts * 1.0 / len(data_loader_train.dataset)
        print("average train DICE is: %.4f"%avg_acc)
        train_loss_list.append(train_loss / len(data_loader_train.dataset))
        train_acc_list.append(avg_acc)

        train_loss_dataframe = pd.DataFrame(data=train_loss_list)
        train_acc_dataframe = pd.DataFrame(data=train_acc_list)
        train_loss_dataframe.to_csv('./output_results/train_loss.csv',index=False)
        train_acc_dataframe.to_csv('./output_results/train_DICE.csv',index=False)
        
        save_interval = 10
        if (epoch + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        
    model.eval()
    for epoch in range(opts.epochs/10):
        print("test_round.............%d"%epoch)
        test_y = []
        test_y_pred = []
        counts=0
        test_loss = 0
        test_batch_num = 0
        outs = []
        labels = []
        with torch.no_grad():
            for test_seq, test_label in data_loader_test:
                test_seq = test_seq.to(device)
                test_label = test_label.to(device)
                test_seq = test_seq.unsqueeze(1)
                
                t_pred = model(test_seq).squeeze(1)
                t_pred = (t_pred-t_pred.min())/(t_pred.max()-t_pred.min())
                outs.append(t_pred)
                labels.append(test_label)
                
                
                
                # accuracy
                loss_bce = bce_loss(t_pred, test_label)
                loss_dice = dice_loss(t_pred, test_label)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
         
                test_loss += loss.item()
                test_batch_num += 1

                

                
                count_cur=dice_coeff(t_pred,test_label)
                print("test-DICE: %.4f"%count_cur)
                counts+=count_cur


        outs = torch.cat(outs,dim=0)
        labels = torch.cat(labels).reshape(-1)
        avg_acc = counts * 1.0 / len(data_loader_test.dataset)
        test_acc_list.append(avg_acc)
        test_loss_list.append(test_loss / len(data_loader_test.dataset))
        print('epoch: %d, test loss: %.4f,test DICE: %.4f' %
              (epoch, test_loss/ test_batch_num,avg_acc))

        test_loss_dataframe = pd.DataFrame(data = test_loss_list)
        test_acc_dataframe = pd.DataFrame(data = test_acc_list)
        test_loss_dataframe.to_csv('./output_results/test_loss.csv',index=False)
        test_acc_dataframe.to_csv('./output_results/test_DICE.csv',index=False)
    






