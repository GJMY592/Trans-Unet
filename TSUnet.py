# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:58:50 2021

@author: GJMY
"""

import torch
from torch import nn
import torch.nn.functional as F
# from TS_layer import Attention
from timm.models.layers import to_2tuple
# import torch.optim as optim

data_size=512

class PatchEmbed(nn.Module):
    def __init__(self, img_size=data_size, patch_size=4, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class Trans_encoder_reshape(nn.Module):
    def __init__(self, patch_reso = [data_size,data_size],heads=8, dim=512,layers = 1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.Trans_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        
        self.Trans_encoder_layers = nn.TransformerEncoder(self.Trans_encoder_layer, num_layers=layers)
        self.patch_reso = patch_reso
        self.up_sample = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4)
    def forward(self, x):
        B,L,C = x.shape
        H,W = self.patch_reso[0], self.patch_reso[1]

        x = self.Trans_encoder_layers(x)
        x = x.transpose(1,2)
        x = x.reshape(B,C,H,W)
        x = self.up_sample(x)
        return x
    
class Trans_block(nn.Module):
    def __init__(self, img_size=[data_size,data_size],patch_size=4,dim=96,heads=8, layers=1):
        super().__init__()
        patch_reso = [int(img_size[0]//patch_size),int(img_size[1]//patch_size)]

        self.Trans_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim)
        self.Trans_encoder = Trans_encoder_reshape(patch_reso, heads, dim, layers)

    def forward(self, x):

        x = self.Trans_embedding(x)
        
        x = self.Trans_encoder(x)
        
        return x
    
class TSU(nn.Module): 
    
    def __init__(self, in_channel=1, out_channel=1): 
        super(TSU, self).__init__() 
        
        # Encode 
        self.conv_encode1 = self.encoder_block(in_channels=in_channel, out_channels=64) 
        self.conv_trans1 = Trans_block([data_size, data_size], patch_size=4, dim=64,
                                      heads=8,layers=1)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2) 
        
        self.conv_encode2 = self.encoder_block(64, 128) 
        self.conv_trans2 = Trans_block([data_size/2, data_size/2], patch_size=4, dim=128,
                                      heads=8,layers=1)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2) 
        
        self.conv_encode3 = self.encoder_block(128, 256) 
        self.conv_trans3 = Trans_block([data_size/4, data_size/4], patch_size=4, dim=256,
                                      heads=8,layers=1)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2) 
        
        self.conv_encode4 = self.encoder_block(256, 512) 
        self.conv_trans4 = Trans_block([data_size/8, data_size/8], patch_size=4, dim=512,
                                      heads=8,layers=1)
        self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
        
        
        # Bottleneck 
        self.bottleneck = torch.nn.Sequential( 
            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(1024), 
            torch.nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(1024), 
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)) 

        # self.conv_trans5 = Trans_block([data_size/16, data_size/16], patch_size=4, dim=1024,
        #                               heads=8,layers=1)
        
        # Decode 
        self.conv_decode4 = self.decoder_block(1024, 512, 256)
        
        self.conv_decode3 = self.decoder_block(512, 256, 128) 
        
        self.conv_decode2 = self.decoder_block(256, 128, 64) 
        
        self.final_layer = self.final_block(128, 64, out_channel)
    
    
    def encoder_block(self, in_channels, out_channels, kernel_size=3): 
        block = torch.nn.Sequential( 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(out_channels), 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(out_channels)) 
        return block  
    
    
    
    def decoder_block(self, in_channels, mid_channel, out_channels, kernel_size=3): 
        block = torch.nn.Sequential( 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(mid_channel), 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(mid_channel), 
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) ) 
        return block  
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3): 
        block = torch.nn.Sequential( 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(mid_channel), 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(mid_channel), 
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1), 
            torch.nn.Sigmoid())
            # torch.nn.BatchNorm2d(out_channels))
        return block  
        
    def crop_and_concat(self, upsampled, bypass, crop=False): 
        if crop: 

            c = (bypass.size()[2] - upsampled.size()[2]) // 2 # Equivalent to floor() 
            bypass = F.pad(bypass, (-c, -c, -c, -c)) 

            return torch.cat((upsampled, bypass), 1)  
        
    def forward(self, x): 
       # Encode 
       encode_block1 = self.conv_encode1(x)
       # encode_block1 = self.conv_trans1(encode_block1)
       encode_pool1 = self.conv_maxpool1(encode_block1)
       
       encode_block2 = self.conv_encode2(encode_pool1) 
       # encode_block2 = self.conv_trans2(encode_block2)
       encode_pool2 = self.conv_maxpool2(encode_block2) 
       
       encode_block3 = self.conv_encode3(encode_pool2) 
       # encode_block3 = self.conv_trans3(encode_block3)
       encode_pool3 = self.conv_maxpool3(encode_block3)
       
       encode_block4 = self.conv_encode4(encode_pool3)
       # encode_block4 = self.conv_trans4(encode_block4)
       encode_pool4 = self.conv_maxpool4(encode_block4) 
       
       # Bottleneck 
       bottleneck1 = self.bottleneck(encode_pool4)
       

       # Decode 
       decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True) 
       cat_layer3 = self.conv_decode4(decode_block4)
       
       decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True) 
       cat_layer2 = self.conv_decode3(decode_block3) 
       
       decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True) 
       cat_layer1 = self.conv_decode2(decode_block2) 
       
       decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True) 
       final_layer = self.final_layer(decode_block1)
       #输出为像素分布在正类的概率
       return final_layer
 



       
     
        
        
        
        