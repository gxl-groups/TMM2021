
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer 



class D_GET_LOGITS(nn.Module):
    def __init__(self, df_ndf, ef_nef):
        super(D_GET_LOGITS, self).__init__()

        self.df_dim = df_ndf # 512
        self.ef_dim = ef_nef # 1024

        out_channel = 1024

        self.interLayer = nn.Sequential(nn.Conv2d(self.df_dim, out_channel, kernel_size=4, stride=4, padding=0), 
                                        nn.BatchNorm2d(out_channel),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=4, padding=0), 
                                        nn.BatchNorm2d(out_channel),
                                        nn.LeakyReLU(0.2, True))

        self.outlogits = nn.Sequential(
                conv3x3(out_channel + self.ef_dim, out_channel),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2, inplace=True),
                conv3x3(out_channel, int(out_channel/2)),
                nn.BatchNorm2d(int(out_channel/2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(out_channel/2), 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code):
        #print('at the begining of forward-----------------')

        inter_h_code = self.interLayer(h_code) # 1024*4*4

        #print(inter_h_code.shape)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        #print(c_code.shape)
        c_code = c_code.repeat(1, 1, 4, 4) # 1024*4*4

        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((inter_h_code, c_code), 1)

        #print(h_c_code.shape)
        #print('at the end of forward-----------------------')
        output = self.outlogits(h_c_code)
        return output.view(-1)


class D_GET_LOGITS_2(nn.Module):
    def __init__(self, df_ndf, ef_nef):
        super(D_GET_LOGITS_2, self).__init__()

        self.df_dim = df_ndf # 512
        self.ef_dim = ef_nef # 1024

        out_channel = 1024

        self.interLayer = nn.Sequential(nn.Conv2d(self.df_dim, out_channel, kernel_size=4, stride=4, padding=0), 
                                        nn.BatchNorm2d(out_channel),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(out_channel, out_channel, kernel_size=2, stride=2, padding=0), 
                                        nn.BatchNorm2d(out_channel),
                                        nn.LeakyReLU(0.2, True))

        self.outlogits = nn.Sequential(
                conv3x3(out_channel + self.ef_dim, out_channel),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2, inplace=True),
                conv3x3(out_channel, int(out_channel/2)),
                nn.BatchNorm2d(int(out_channel/2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(out_channel/2), 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code):
        #print('at the begining of forward-----------------')

        inter_h_code = self.interLayer(h_code) # 1024*4*4

        #print(inter_h_code.shape)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        #print(c_code.shape)
        c_code = c_code.repeat(1, 1, 4, 4) # 1024*4*4

        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((inter_h_code, c_code), 1)

        #print(h_c_code.shape)
        #print('at the end of forward-----------------------')
        output = self.outlogits(h_c_code)
        return output.view(-1)


class MultiscaleDiscriminator(nn.Module):
        # print('the params of D--------------------------------------------')
        # print(netD_input_nc)            # 39; 6
        # print(opt.ndf)                  # 64; 64
        # print(opt.n_layers_D)           # 3; 3
        # print(opt.norm)                 # instance; instance
        # print(use_sigmoid)              # False; False
        # print(opt.num_D)                # 2; 2
        # print(opt.no_ganFeat_loss)      # False; False
        # print(self.gpu_ids)             # [0]; [0]
        # print('the end of params--------------------------------------')
        
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.get_cond_logits = D_GET_LOGITS(512, 1024)
        self.get_cond_logits_2 = D_GET_LOGITS_2(512, 1024)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):  
        # print('in side D forward---------------------')
        # print(input.shape) # (2, 6, 512, 512)
       
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)

        # print(result[0][0].shape) # (2, 64, 257, 257)
        # print(result[0][1].shape) # (2, 128, 129, 129)
        # print(result[0][2].shape) # (2, 256, 65, 65)
        # print(result[0][3].shape) # (2, 512, 66, 66)
        # print(result[0][4].shape) # (2, 1, 67, 67)

        # print(result[1][0].shape) # (2, 64, 129, 129)
        # print(result[1][1].shape) # (2, 128, 65, 65)
        # print(result[1][2].shape) # (2, 256, 33, 33)
        # print(result[1][3].shape) # (2, 512, 34, 34)
        # print(result[1][4].shape) # (2, 1, 35, 35)
        # print('end of D forward---------------------')
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        





from torchsummary import summary

# netG = GlobalGenerator(3, 3)
# if torch.cuda.is_available():
#     netG.cuda()
# print(netG)


netD = MultiscaleDiscriminator(6, 64, 3, get_norm_layer(), False, 2, False) 
if torch.cuda.is_available():
    netD.cuda()

summary(netD, input_size=(6, 512, 512)) 
print(netD)


# summary(netG, input_size=(3, 512, 512)) 




