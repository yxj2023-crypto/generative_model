import torch
import torch.nn as nn 


class ConvModule(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 embed_dim,
                 
                 ):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.embed_fusion=nn.Linear(embed_dim,in_channel)
        self.shortcut=nn.Conv2d(in_channel,out_channel,kernel_size=1)if in_channel != out_channel else nn.Identity()
        
    def forward(self,x,embed):
        # x:(B,C,H,W),embed:(B,C')
        x_embed=x+self.embed_fusion(embed)[:,:,None,None]
        conv_out=self.conv_layers(x_embed)
        # 原项目原地加入编码，残差分支也使用加入编码后的输入。
        res_out=conv_out+self.shortcut(x_embed)
        return res_out

class Unet(nn.Module):
    def __init__(self,input_channel=1,base_channel=64,embed_dim=64):
        super().__init__()
        if embed_dim <= 0 or embed_dim % 2:
            raise ValueError('embed_dim must be a positive even number')
        self.embed_dim=embed_dim
        self.conv_in=nn.Conv2d(input_channel,base_channel,kernel_size=3,padding=1)
        self.down1=nn.ModuleList(
            [
                ConvModule(base_channel,base_channel*2,embed_dim),
                ConvModule(base_channel*2,base_channel*2,embed_dim),
             ]
        )
        self.pool=nn.MaxPool2d(2)
        self.down2=nn.ModuleList(
            [
                ConvModule(base_channel*2,base_channel*4,embed_dim),
                ConvModule(base_channel*4,base_channel*4,embed_dim),
             ]
        )
        self.mid_layer=ConvModule(base_channel*4,base_channel*4,embed_dim)

        self.upsample=nn.Upsample(scale_factor=2)
        self.up1=nn.ModuleList(
            [
                ConvModule(base_channel*4*2,base_channel*2,embed_dim),
                ConvModule(base_channel*2,base_channel*2,embed_dim),
             ]
        )
        self.up2=nn.ModuleList(
            [
                ConvModule(base_channel*2*2,base_channel,embed_dim),
                ConvModule(base_channel,base_channel,embed_dim),
             ]
        )
        self.conv_out=nn.Conv2d(base_channel,input_channel,kernel_size=1)
    def _sinusoidal_emb(self,input):
        # 将标量进行正弦编码
        input = input * 1000

        freqs = torch.pow(10000, torch.linspace(0, 1, self.embed_dim // 2)).to(input.device)
        sin_emb = torch.sin(input[:, None] / freqs)
        cos_emb = torch.cos(input[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)
    def forward(self,x,t,y=None):
        
        x=self.conv_in(x)
        embed=self._sinusoidal_emb(input=t)
        if y is not None:
            y_embed=self._sinusoidal_emb(input=y)
            # label编码，-1表示无条件生成，仅用于训练区分，推理的时候不需要
            # 把y等于-1的index找出来，然后把对应的y_emb设置为0
            y_embed[y == -1] = 0.0
            embed+=y_embed 

        # down layer 1：
        for layer in self.down1:
            x=layer(x,embed)
        x1=x  # 28x28
        x=self.pool(x)
        # --?

        # down layer 2:
        for layer in self.down2:
            x=layer(x,embed)
        x2=x  # 14x14
        x=self.pool(x)
        # --?

        # mid layer
        x=self.mid_layer(x,embed)  # 7x7
        # --?

        # up layer 1:
        x=torch.cat([self.upsample(x),x2],dim=1)
        for layer in self.up1:
            x=layer(x,embed)
        # --?

        # up layer 2:
        x=torch.cat([self.upsample(x),x1],dim=1)
        for layer in self.up2:
            x=layer(x,embed)
        # --?
        return self.conv_out(x)
