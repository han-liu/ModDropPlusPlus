import torch

from models.tiramisu_layers_dyn import *


class FCDenseNetDyn(nn.Module):
    # copied from https://github.com/bfortuner/pytorch_tiramisu
    # slightly changed to output range from -1 to 1
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))

        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count, growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.tanh = nn.Tanh()
        # self.softmax = nn.LogSoftmax(dim=1)

        # self.controller = nn.Conv2d(5, 15*3*3*48+48, kernel_size=1, stride=1, padding=0)
        self.controller = nn.Conv2d(5, 15*48 + 48, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mc, get_dyn_feat=False):  # add modality code as additional input
        # adapted from https://github.com/jianpengz/DoDNet/blob/main/a_DynConv/unet3D_DynConv882.py

        N, _, H, W = x.size()
        x = x.reshape(1, -1, H, W)  # 1 x N*15 x 128 x 128

        params = self.controller(mc.unsqueeze(-1).unsqueeze(-1).float())  # 1 x 48*15+48 x 1 x 1
        params.squeeze_(-1).squeeze_(-1)  # 1 x 48*15+48
        # print(params.size())  # N x 768
        weight_nums, bias_nums = [], []
        weight_nums.append(15 * 48)
        bias_nums.append(48)
        sw, sb = parse_dynamic_params(params, weight_nums, bias_nums)
        sw = sw.reshape(N, 48, 15).unsqueeze(-1).unsqueeze(-1)
        sb = sb.reshape(N, 48)

        """generate all parameters: 5 -> 6000 mapping"""
        # N, _, H, W = x.size()
        # x = x.reshape(1, -1, H, W)
        # out = my_first_conv(x, weights, biases, N)
        # out = out.reshape(-1, 48, out.size()[-2], out.size()[-1])

        w, b = self.firstconv.weight.clone().unsqueeze(0), self.firstconv.bias.clone()  # 1 x 48 x 15 x 3 x 3
        # This function is differentiable, so gradients will flow back from the result of this operation to input.
        # To create a tensor without an autograd relationship to input see detach()

        # check if both scalars and kernels are updated!
        # print('======================================')
        # print(self.firstconv.weight[0, 0, :, :], self.firstconv.bias[0])
        # print(sw[0, :3, :3, 0, 0])

        w = w * sw
        b = b * sb

        # print(sw.size(), w.size())
        # print(w.size(), b.size())

        w = w.reshape(-1, 15, 3, 3)
        b = b.reshape(-1)

        out = dynamic_head(x, w, b, N)
        out = out.reshape(-1, 48, out.size()[-2], out.size()[-1])

        # dynamic filter place! #
        if get_dyn_feat:
            dyn_feat = out

        # regular network
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        # print(out)
        out = self.bottleneck(out)

        # BOTTLENECK PLACE! # deprecated
        # print('===============')
        # if get_dyn_feat:
        #     dyn_feat = out
        # print('===============')

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.tanh(out)
        # out = self.softmax(out)
        # print(out.size())
        
        if get_dyn_feat:
            return dyn_feat, out
        return out


if __name__ == '__main__':
    from torchsummary import summary
    net = FCDenseNetDyn(in_channels=15, down_blocks=(4, 4, 4, 4, 4), up_blocks=(4, 4, 4, 4, 4),
                        bottleneck_layers=4, growth_rate=12, out_chans_first_conv=48, n_classes=1).cuda()

    data = torch.zeros((1, 15, 128, 128)).cuda()
    mc = torch.zeros((1, 5)).cuda()
    output = net(data, mc)
    print('finish running...')

