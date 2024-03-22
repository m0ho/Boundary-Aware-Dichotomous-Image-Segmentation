
import torch
import torch.nn as nn
import torch.nn.functional as F
from Res import resnet18
from Swin import Swintransformer
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math
from ResNet_models_Custom import Triple_Conv, multi_scale_aspp, Classifier_Module, RCAB, BasicConv2d
Act = nn.ReLU

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out
    def initialize(self):
        weight_init(self)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    def initialize(self):
        weight_init(self)
        
class Pyramid_block(nn.Module):
    def __init__(self, in_channels, in_resolution,out_channels,out_resolution,heads,initial):
        super(Pyramid_block, self).__init__()
        self.block1 = nn.ModuleList()
        if in_channels != out_channels:
            self.block1.append(Triple_Conv(in_channels, out_channels))
        if initial==1:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(out_channels, width=in_resolution, height=in_resolution, heads=heads))
        elif initial==2:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(in_channels, width=in_resolution, height=in_resolution, heads=heads))
        elif initial==3:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(in_channels, width=in_resolution, height=in_resolution, heads=heads))
        elif initial==4:
            self.block1.append(multi_scale_aspp(in_channels))
        self.block1 = nn.Sequential(*self.block1)
        self.in_resolution = in_resolution
        self.out_resolution = out_resolution

    def forward(self, x):
        x = self.block1(x)
        if self.in_resolution != self.out_resolution:
            x = F.interpolate(x, size=(self.out_resolution,self.out_resolution), mode='bilinear',align_corners=True)
        return x
    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax,nn.PReLU,nn.Dropout2d,nn.Sigmoid)):
            pass
        else:
            m.initialize()

    
class Grafting(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y):
        batch_size = x.shape[0]
        chanel     = x.shape[1]
        sc = x
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        sc1 = x
        x = self.lnx(x)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)
        y = self.lny(y)
        
        B, N, C = x.shape
        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_qv= self.qv(x).reshape(B,N,2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1] 
        y_k = y_k[0]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = (x+sc1)

        x = x.permute(0,2,1)
        x = x.view(batch_size,chanel,*sc.size()[2:])
        x = self.conv2(x)+x
        return x,self.act(self.bn(self.conv(attn+attn.transpose(-1,-2))))


    def initialize(self):
        weight_init(self)
        

        
class DB1(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB1,self).__init__()
        self.squeeze1 = nn.Sequential(  
                    nn.Conv2d(inplanes, outplanes,kernel_size=1,stride=1,padding=0), 
                    nn.BatchNorm2d(64), 
                    nn.ReLU(inplace=True)
                )
        self.squeeze2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3,stride=1,dilation=2,padding=2), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        z = self.squeeze2(self.squeeze1(x))   
        return z,z

    def initialize(self):
        weight_init(self)

class DB2(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB2,self).__init__()
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes+outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,z): 
        z = F.interpolate(z,size=x.size()[2:],mode='bilinear',align_corners=True)
        p = self.conv(torch.cat((x,z),1))
        sc = self.short_cut(z)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p,p
    
    def initialize(self):
        weight_init(self)

class DB3(nn.Module):
    def __init__(self) -> None:
        super(DB3,self).__init__()

        self.db2 = DB2(64,64)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.sqz_s1=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
    def forward(self,s,r,up):
        up = F.interpolate(up,size=s.size()[2:],mode='bilinear',align_corners=True)
        s = self.sqz_s1(s)
        r = self.sqz_r4(r)
        sr = self.conv3x3(s+r)
        out,_  =self.db2(sr,up)
        return out,out
    def initialize(self):
        weight_init(self)
        
class SplitConvBlock(nn.Module):
    def __init__(self, channel, scales):
        super(SplitConvBlock, self).__init__()
        self.scales = scales
        self.width = math.ceil(channel/scales)
        self.channel1 = self.width
        self.channel2 = self.width + self.channel1//2
        self.channel3 = self.width + self.channel2//2
        self.channel4 = self.width + self.channel3//2
        self.channel5 = self.width + self.channel4//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False), nn.BatchNorm2d(self.channel1), nn.PReLU()
        )
        if scales > 2:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.channel2, self.channel2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False), nn.BatchNorm2d(self.channel2), nn.PReLU()
            )
        if scales > 3:
            self.conv3 = nn.Sequential(
                nn.Conv2d(self.channel3, self.channel3, kernel_size=3, stride=1, padding=4, dilation=4, bias=False), nn.BatchNorm2d(self.channel3), nn.PReLU()
            )
        if scales > 4:
            self.conv4 = nn.Sequential(
                nn.Conv2d(self.channel4, self.channel4, kernel_size=3, stride=1, padding=6, dilation=6, bias=False), nn.BatchNorm2d(self.channel4), nn.PReLU()
            )
        if scales > 5:
            self.conv5 = nn.Sequential(
                nn.Conv2d(self.channel5, self.channel5, kernel_size=3, stride=1, padding=8, dilation=8, bias=False), nn.BatchNorm2d(self.channel5), nn.PReLU()
            )

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        sp1 = self.conv1(spx[0])

        if self.scales > 2:
            sp1x = torch.split(sp1, math.ceil(self.channel1/2), 1)
            sp2 = torch.cat((spx[1], sp1x[1]), 1)
            sp2 = self.conv2(sp2)

        if self.scales > 3:
            sp2x = torch.split(sp2, math.ceil(self.channel2/2), 1)
            sp3 = torch.cat((spx[2], sp2x[1]), 1)
            sp3 = self.conv3(sp3)

        if self.scales > 4:
            sp3x = torch.split(sp3, math.ceil(self.channel3/2), 1)
            sp4 = torch.cat((spx[3], sp3x[1]), 1)
            sp4 = self.conv4(sp4)

        if self.scales > 5:
            sp4x = torch.split(sp4, math.ceil(self.channel4/2), 1)
            sp5 = torch.cat((spx[4], sp4x[1]), 1)
            sp5 = self.conv5(sp5)

        if self.scales == 1:
            x = sp1
        elif self.scales == 2:
            x = torch.cat((sp1, spx[1]), 1)
        elif self.scales == 3:
            x = torch.cat((sp1x[0], sp2, spx[2]), 1)
        elif self.scales == 4:
            x = torch.cat((sp1x[0], sp2x[0], sp3, spx[3]), 1)
        elif self.scales == 5:
            x = torch.cat((sp1x[0], sp2x[0], sp3x[0], sp4, spx[4]), 1)
        elif self.scales == 6:
            x = torch.cat((sp1x[0], sp2x[0], sp3x[0], sp4x[0], sp5, spx[5]), 1)

        return x

    def initialize(self):
        weight_init(self)
                
class BA1(nn.Module):
    def __init__(self) -> None:
        super(BA1,self).__init__()

        self.db2 = DB2(64,64)

        self.HSC = SplitConvBlock(64, 6)
        
        self.edge1 = nn.Conv2d(64, 1, 3, padding=1)
        
        self.edge2 = nn.Conv2d(64, 1, 3, padding=1)
        
        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(256+32, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.sqz_s1=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
    def forward(self,s,r,up):
        up = F.interpolate(up,size=s.size()[2:],mode='bilinear',align_corners=True)
        s = self.sqz_s1(s)
        r = self.sqz_r4(r)
        sr = self.HSC(s+r)##
        out,_  =self.db2(sr,up)
        e   = self.edge1(sr)##
        out = self.edge2(out)
        return out,e
    def initialize(self):
        weight_init(self)

class BA2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BA2, self).__init__()
        self.convert = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
        )
        self.conve1 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.conve2 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.conve3 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.conve4 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.edge1 = nn.Conv2d(out_channel, 1, 3, padding=1)
        self.edge2 = nn.Conv2d(out_channel, 1, 3, padding=1)
        self.convr = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
            nn.Conv2d(out_channel, 1, 3, padding=1)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )

    def forward(self, x, y, e):
        x = self.convert(x)
        xe = self.conve1(torch.cat((x, y), 1))
        a = -1*torch.sigmoid(y) + 1
        x = a.expand_as(x).mul(x)

        y1 = self.conve2(torch.cat((x, y), 1))
        e0 = self.conve3(torch.cat((x, y), 1))#e0 is y1
        e1 = self.conve4(torch.cat((e0, e), 1))
        y2 = self.branch1(torch.cat((e1, y1), 1))
        e2 = self.branch2(torch.cat((e1, y2), 1))
        e = self.edge1(e2) + e
        y = self.edge2(y2) + y

        return y, e

    def initialize(self):
        weight_init(self)

class decoder_BG(nn.Module):
    def __init__(self) -> None:
        super(decoder_BG,self).__init__()
        self.sqz_s2=nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
        self.sqz_r5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.GF   = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
        
        self.d1 = DB1(512,64)
        self.d2 = DB2(512,64)
        self.d3 = DB2(64,64)
        self.d4 = BA1()
        self.d5 = BA2(128+32,64)
        self.d6 = BA2(64+32,32)

    def forward(self,s1,s2,s3,s4,r2,r3,r4,r5):
        r5 = F.interpolate(r5,size = s2.size()[2:],mode='bilinear',align_corners=True) 
        s1 = F.interpolate(s1,size = r4.size()[2:],mode='bilinear',align_corners=True) 

        s4_,_ = self.d1(s4)
        s3_,_ = self.d2(s3,s4_)

        s2_ = self.sqz_s2(s2)
        r5_= self.sqz_r5(r5)

        graft_feature_r5 = self.GF(torch.cat((s2_, r5_), 1))
        graft_feature_r5_,_=self.d3(graft_feature_r5,s3_)

        y4,e4 =self.d4(s1,r4,graft_feature_r5_)
        y4_3   = F.interpolate(y4, r3.size()[2:], mode='bilinear', align_corners=True)
        e4_3   = F.interpolate(e4, r3.size()[2:], mode='bilinear', align_corners=True)

        y3,e3 = self.d5(r3, y4_3, e4_3)
        e3_2   = F.interpolate(e3, r2.size()[2:], mode='bilinear', align_corners=True)
        y3_2   = F.interpolate(y3, r2.size()[2:], mode='bilinear', align_corners=True)
        	
        y2,e2 = self.d6(r2, y3_2, e3_2)

        return y2,e2,y3,e3,y4,e4
        
    def initialize(self):
        weight_init(self)

class FPT(nn.Module):
    def __init__(self) -> None:
        super(FPT,self).__init__()
        
        self.sqz512 = nn.Sequential(
            nn.Conv2d(512, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU()
        )
        self.sqz256 = nn.Sequential(
            nn.Conv2d(256, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU()
        )
        self.sqz128 = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU()
        )
        self.sqz64 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU()
        )
        self.sqz64_ = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU()
        )
        ##my_FPSA
        self.mhsa5    = MHSA(32, width=12, height=12, heads=4)
        self.daspp5_1 = multi_scale_aspp(64)
        self.selayer5 = SELayer(channel=32, reduction=16)
        self.conv_se5 = Triple_Conv(64, 32)
        
        self.mhsa4    = MHSA(32, width=24, height=24, heads=4)
        self.daspp4_1 = multi_scale_aspp(64)
        self.daspp4_2 = multi_scale_aspp(96)
        self.selayer4 = SELayer(channel=32, reduction=16)
        self.conv_se4 = Triple_Conv(96, 32)
        
        self.mhsa3    = MHSA(32, width=48, height=48, heads=4)
        self.daspp3_1 = multi_scale_aspp(64)
        self.daspp3_2 = multi_scale_aspp(96)
        self.selayer3 = SELayer(channel=32, reduction=16)
        self.conv_se3 = Triple_Conv(96, 32)
        
        self.mhsa2    = MHSA(32, width=48, height=48, heads=4)
        self.daspp2_1 = multi_scale_aspp(64)
        self.daspp2_2 = multi_scale_aspp(96)
        self.selayer2 = SELayer(channel=32, reduction=16)
        self.conv_se2 = Triple_Conv(96, 32)

    def forward(self,input_r2,input_r3,input_r4,input_r5):

        r2_shape = input_r2.size()[2:]
        r3_shape = input_r3.size()[2:]
        r4_shape = input_r4.size()[2:]
        r5_shape = input_r5.size()[2:]

        r5_rc = self.sqz512(input_r5)
        r4_rc = self.sqz256(input_r4)
        r3_rc = self.sqz128(input_r3)
        r2_rc = self.sqz64 (input_r2)

        r5_rc_rs = F.interpolate(r5_rc, size=(12,12), mode='bilinear',align_corners=True)
        r4_rc_rs = F.interpolate(r4_rc, size=(24,24), mode='bilinear',align_corners=True)
        r3_rc_rs = F.interpolate(r3_rc, size=(48,48), mode='bilinear',align_corners=True)
        r2_rc_rs = F.interpolate(r2_rc, size=(48,48), mode='bilinear',align_corners=True)
        
        r5_rc_rs_ = self.mhsa5(r5_rc_rs)
        r4_rc_rs_ = self.mhsa4(r4_rc_rs)
        r3_rc_rs_ = self.mhsa3(r3_rc_rs)    
        r2_rc_rs_ = self.mhsa2(r2_rc_rs)
        
        r5_rc_rs_5_1 = self.daspp5_1(torch.cat((r5_rc_rs_, r5_rc_rs), 1))
        r5_rc_rs_se = self.conv_se5(r5_rc_rs_5_1)
        r5_rc_rs_SE = self.selayer5(r5_rc_rs_se)
        r5_rc_rs_r4 = F.interpolate(r5_rc_rs_SE, size=r4_rc_rs.size()[2:], mode='bilinear',align_corners=True)
        
        r4_rc_rs_4_1 = self.daspp4_1(torch.cat((r4_rc_rs_, r5_rc_rs_r4), 1))
        r4_rc_rs_4_2 = self.daspp4_2(torch.cat((r4_rc_rs_4_1, r4_rc_rs), 1))
        r4_rc_rs_se = self.conv_se4(r4_rc_rs_4_2)
        r4_rc_rs_SE = self.selayer4(r4_rc_rs_se)
        r4_rc_rs_r3 = F.interpolate(r4_rc_rs_SE, size=r3_rc_rs.size()[2:], mode='bilinear',align_corners=True)
        
        r3_rc_rs_3_1 = self.daspp3_1(torch.cat((r3_rc_rs_, r4_rc_rs_r3), 1))
        r3_rc_rs_3_2 = self.daspp3_2(torch.cat((r3_rc_rs_3_1, r3_rc_rs), 1))
        r3_rc_rs_se = self.conv_se3(r3_rc_rs_3_2)
        r3_rc_rs_SE = self.selayer3(r3_rc_rs_se)
        r3_rc_rs_r2 = F.interpolate(r3_rc_rs_SE, size=r2_rc_rs.size()[2:], mode='bilinear',align_corners=True)
        
        r2_rc_rs_2_1 = self.daspp2_1(torch.cat((r2_rc_rs_, r3_rc_rs_r2), 1))
        r2_rc_rs_2_2 = self.daspp2_2(torch.cat((r2_rc_rs_2_1, r2_rc_rs), 1))
        r2_rc_rs_se = self.conv_se2(r2_rc_rs_2_2)
        r2_rc_rs_SE = self.selayer2(r2_rc_rs_se)

        conv54 = F.interpolate(r4_rc_rs_SE, size=r4_shape, mode='bilinear',align_corners=True)
        conv543 = F.interpolate(r3_rc_rs_SE, size=r3_shape, mode='bilinear',align_corners=True)
        conv5432 = F.interpolate(r2_rc_rs_SE, size=r2_shape, mode='bilinear',align_corners=True)
        
        r4_ = torch.cat((conv54,input_r4), 1)
        r3_ = torch.cat((conv543,input_r3), 1)
        r2_ = torch.cat((conv5432,input_r2), 1)

        return r2_,r3_,r4_ 
        
    def initialize(self):
        weight_init(self)


class BADIS(nn.Module):
    def __init__(self, cfg=None):
        super(BADIS, self).__init__()
        self.cfg      = cfg
        self.test_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.decoder  = decoder_BG()
        self.myFPSA = FPT()
        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        self.resnet    = resnet18()
        self.swin      = Swintransformer(224)
        self.swin.load_state_dict(torch.load('../pre/swin224.pth')['model'],strict=False)
        # self.resnet.load_state_dict(torch.load('../pre/resnet18.pth'),strict=False)
        
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain=torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k,v in pretrain.items():
                new_state_dict[k[7:]] = v  
            self.load_state_dict(new_state_dict, strict=False)  

    def forward(self, x,shape=None,mask=None):
        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(224,224), mode='bilinear',align_corners=True)

        r2, r3, r4, r5 = self.resnet(x)
        s1,s2,s3,s4 = self.swin(y)
        r2_,r3_,r4_ = self.myFPSA(r2, r3, r4, r5)
        y2,e2,y3,e3,y4,e4= self.decoder(s1,s2,s3,s4,r2_,r3_,r4_,r5)

        y2 = F.interpolate(y2, size=shape, mode='bilinear') 
        e2 = F.interpolate(e2, size=shape, mode='bilinear') 
        y3 = F.interpolate(y3, size=shape, mode='bilinear') 
        e3 = F.interpolate(e3, size=shape, mode='bilinear') 
        y4 = F.interpolate(y4, size=shape, mode='bilinear') 
        e4 = F.interpolate(e4, size=shape, mode='bilinear') 

        return y2,[y2,y3,y4,e2,e3,e4]


    

