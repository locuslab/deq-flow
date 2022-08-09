import torch
import torch.nn as nn
import torch.nn.functional as F

from gma import Aggregate


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class MotionEncoder(nn.Module):
    def __init__(self, args):
        super(MotionEncoder, self).__init__()
        
        if args.large:
            c_dim_1 = 256 + 128
            c_dim_2 = 192 + 96

            f_dim_1 = 128 + 64
            f_dim_2 = 64 + 32

            cat_dim = 128 + 64
        elif args.huge:
            c_dim_1 = 256 + 256
            c_dim_2 = 192 + 192

            f_dim_1 = 128 + 128
            f_dim_2 = 64 + 64

            cat_dim = 128 + 128
        elif args.gigantic:
            c_dim_1 = 256 + 384
            c_dim_2 = 192 + 288

            f_dim_1 = 128 + 192
            f_dim_2 = 64 + 96

            cat_dim = 128 + 192
        elif args.tiny:
            c_dim_1 = 64
            c_dim_2 = 48

            f_dim_1 = 32
            f_dim_2 = 16

            cat_dim = 32
        else:
            c_dim_1 = 256
            c_dim_2 = 192

            f_dim_1 = 128
            f_dim_2 = 64

            cat_dim = 128

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, c_dim_1, 1, padding=0)
        self.convc2 = nn.Conv2d(c_dim_1, c_dim_2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, f_dim_1, 7, padding=3)
        self.convf2 = nn.Conv2d(f_dim_1, f_dim_2, 3, padding=1)
        self.conv = nn.Conv2d(c_dim_2+f_dim_2, cat_dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(UpdateBlock, self).__init__()
        self.args = args
    
        if args.tiny:
            cat_dim = 32
        elif args.large:
            cat_dim = 128 + 64
        elif args.huge:
            cat_dim = 128 + 128
        elif args.gigantic:
            cat_dim = 128 + 192
        else:
            cat_dim = 128
        
        if args.old_version:
            flow_head_dim = min(256, 2*cat_dim)
        else:
            flow_head_dim = 2*cat_dim

        self.encoder = MotionEncoder(args)
        
        if args.gma:
            self.gma = Aggregate(dim=cat_dim, dim_head=cat_dim, heads=1)

            gru_in_dim = 2 * cat_dim + hidden_dim
        else:
            self.gma = None

            gru_in_dim = cat_dim + hidden_dim
        
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=gru_in_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=flow_head_dim)
            
    def forward(self, net, inp, corr, flow, attn=None, upsample=True):
        motion_features = self.encoder(flow, corr)
        
        if self.gma:
            motion_features_global = self.gma(attn, motion_features)
            inp = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)
        
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, delta_flow



