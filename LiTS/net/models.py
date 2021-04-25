# 2dKiUnet
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F

class kiunet_org(nn.Module):
    def __init__(self,in_ch, out_ch,training):
        super(kiunet_org, self).__init__()
        self.training = training
        # self.start = nn.Conv2d(in_ch, 1, 3, stride=2, padding=1)

        # unet
        self.encoder1 = nn.Conv2d(in_ch, 32, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.en1_bn = nn.BatchNorm2d(32)
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  
        self.en2_bn = nn.BatchNorm2d(64)
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(128)

        self.decoder1 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)   
        self.de1_bn = nn.BatchNorm2d(128)
        self.decoder2 =   nn.Conv2d(128,64, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(64)
        self.decoder3 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(32)

        # kiunet
        self.encoderf1 =   nn.Conv2d(in_ch, 32, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.enf1_bn = nn.BatchNorm2d(32)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(64)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(128)

        self.decoderf1 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(128)
        self.decoderf2=   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(64)
        self.decoderf3 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(32)

        self.intere1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(32)
        self.intere2_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(64)
        self.intere3_1 = nn.Conv2d(128,128,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(128)

        self.intere1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(32)
        self.intere2_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(64)
        self.intere3_2 = nn.Conv2d(128,128,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(128)

        self.interd1_1 = nn.Conv2d(128,128,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(128)
        self.interd2_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(64)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(128,128,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(128)
        self.interd2_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(64)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        # self.start = nn.Conv3d(1, 1, 3, stride=1, padding=1)
        self.final = nn.Conv2d(32,out_ch,1,stride=1,padding=0)
        self.fin = nn.Conv2d(out_ch,out_ch,1,stride=1,padding=0)

        # 256*256 尺度下的映射
        self.map4 = nn.Sequential(
            nn.Conv2d(32, out_ch, 1, 1),
            # nn.Upsample(scale_factor=(8, 8), mode='bilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv2d(32, out_ch, 1, 1),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv2d(64, out_ch, 1, 1),
            nn.Upsample(scale_factor=(4, 4), mode='bilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv2d(128, out_ch, 1, 1),
            nn.Upsample(scale_factor=(8, 8), mode='bilinear'),
            nn.Sigmoid()
        )
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x),2,2)))  #U-Net branch 32 H/2 W/2
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2, 2),mode ='bilinear'))) #Ki-Net branch 32 2H 2W
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bilinear')) #CRFB
        u1 = out  # for skip conn
        o1 = out1  # for skip conn

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.25*0.25, 0.25*0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))

        u2 = out
        o2 = out1
        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.0625*0.25, 0.0625*0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bilinear'))

        ### Start Decoder
        out = F.relu(self.de1_bn(self.decoder1(out)))  #U-NET 
        out1 = F.relu(self.def1_bn(self.decoderf1(out1))) #Ki-NET
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625*0.25,0.0625*0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(64,64),mode ='bilinear'))
        # 先经过crbf再去skip connect
        output1 = self.map1(out)

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(2,2))))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1),2,2)))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))

        out = torch.add(out,u2)
        out1 = torch.add(out1,o2)
        output2 = self.map2(out) ####deep supervision
        
        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1),2,2)))
        output3 = self.map3(out)
        out = torch.add(out,u1)  #skip conn
        out1 = torch.add(out1,o1)  #skip conn
        out = F.interpolate(out, scale_factor=(2, 2), mode = "bilinear")
        out1 = F.max_pool2d(out1, 2, 2)


        out = torch.add(out,out1) # fusion of both branches
        out = F.relu(self.final(out))  #1*1 conv
        # print(out.shape) # 1 2 256 256
        
        output4 = out
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
        
# def init(module):
#     if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
#         nn.init.kaiming_normal_(module.weight.data, 0.25)
#         nn.init.constant_(module.bias.data, 0)
if __name__ == "__main__":
    model = kiunet_org(in_ch = 3, out_ch = 2, training=True).cuda()
    img = torch.randn(1, 3, 256, 256).cuda()
    output = model(img)
    print(output[3].shape)
    print('net total parameters:', sum(param.numel() for param in model.parameters()))

