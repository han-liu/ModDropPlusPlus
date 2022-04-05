import torch
import pytorch_ssim
import random
import numpy as np
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from .losses import FocalLoss, dice_loss
from . import networks
from configurations import *


class MsModel(BaseModel): # ModDrop++ model
    def name(self):
        return 'MsModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L2', type=int, default=2000, help='weight for L2 loss') 
            parser.add_argument('--lambda_dice', type=int, default=100, help='weight for dice loss')
            parser.add_argument('--lambda_SSIM', type=int, default=200, help='weight for SSIM loss (feature space)')
            parser.add_argument('--lambda_KL', type=int, default=100, help='weight for KL loss (feature space)')
            parser.add_argument('--lambda_focal', type=int, default=10000, help='weight for focal loss')
        return parser

    def initialize(self, opt, model_suffix):
        BaseModel.initialize(self, opt, model_suffix)
        self.isTrain = opt.isTrain
        self.use_dyn = opt.use_dyn
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc * len(MODALITIES), opt.init_type, opt.init_gain, self.gpu_ids, dyn=self.use_dyn)  # Network input 3 slices x 5 modalities
        
        if self.isTrain:
            self.visual_names = ['real_mask', 'full_mask', 'miss_mask']
            for modality in MODALITIES:
                self.visual_names += ['full_' + modality]
                self.visual_names += ['miss_' + modality]
        else:
            self.visual_names = ['real_mask', 'fake_mask']
            for modality in MODALITIES:
                self.visual_names += [modality]
                
        if self.isTrain:
            self.loss_names = ['total']
            self.criterion_names = []
            criterions = {'L2': torch.nn.MSELoss(), 'focal': FocalLoss(gamma=1, alpha=0.25).to(self.device), 'dice': dice_loss, 'ssim': pytorch_ssim.SSIM(), 'KL': torch.nn.KLDivLoss()}
            for k in criterions.keys():
                if k in opt.loss_to_use:
                    self.loss_names += [k]
                    setattr(self, 'criterion_%s' % k, criterions[k])
                    self.criterion_names.append(k)
            assert len(self.criterion_names), 'should use at least one loss function in L2, focal, dice'
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # for feature extraction, update only the last layer, otherwise update all the parameters
            if self.opt.feature_extract:
                params_to_update = []
                print("Params to learn:")
                for name, param in self.netG.named_parameters():
                    if 'thres' in name:
                        params_to_update.append(param)
                    else:
                        param.requires_grad = False
            else:
                params_to_update = self.netG.parameters()
            self.optimizers = [torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999))]

    def set_input(self, input):
        if self.isTrain:  # training phase
            data_full, data_miss = input
            self.full_dc = data_full['dc'].to(self.device)
            self.full_mc = data_full['mc'].to(self.device)
            self.miss_dc = data_miss['dc'].to(self.device)
            self.miss_mc = data_miss['mc'].to(self.device)
            self.real_mask = data_full['mask'].to(self.device)
            for modality in MODALITIES:
                setattr(self, 'full_' + modality, data_full[modality].to(self.device))
                setattr(self, 'miss_' + modality, data_miss[modality].to(self.device))
            self.full_input = torch.cat([getattr(self, 'full_' + k) for k in MODALITIES], 1)
            self.miss_input = torch.cat([getattr(self, 'miss_' + k) for k in MODALITIES], 1)
        else:  # inference phase
            self.mc = input['mc'].to(self.device)
            for modality in MODALITIES:
                setattr(self, modality, input[modality].to(self.device))
            self.real_mask = input['mask'].to(self.device)
            self.input = torch.cat([getattr(self, k) for k in MODALITIES], 1)

    def forward(self):
        if self.isTrain:
            self.full_feat, self.full_mask = self.netG(self.full_input, self.full_mc, get_dyn_feat=True)
            self.miss_feat, self.miss_mask = self.netG(self.miss_input, self.miss_mc, get_dyn_feat=True)
        else:
            self.fake_mask = self.netG(self.input, self.mc)  # inference phase


    def backward_G(self):
        def normalize(feat):
            vector = torch.flatten(feat)
            min_v = torch.min(vector)
            range_v = torch.max(vector) - min_v
            if range_v > 0:
                normalised = (vector - min_v) / range_v
            else:
                normalised = torch.zeros(vector.size())
            return normalised

        self.loss_total = 0
        full_mask = (self.full_mask + 1) / 2
        miss_mask = (self.miss_mask + 1) / 2
        real_mask = (self.real_mask + 1) / 2
        for k, criterion_name in enumerate(self.criterion_names):
            criterion = getattr(self, 'criterion_%s' % criterion_name)
            if criterion_name == 'dice':
                cur_loss = criterion(full_mask, real_mask) * self.opt.lambda_dice + criterion(miss_mask, real_mask) * self.opt.lambda_dice
                self.loss_total += cur_loss
            elif self.criterion_names[k] == 'focal':
                cur_loss = criterion(self.full_mask, real_mask) * self.opt.lambda_focal + criterion(self.miss_mask, real_mask) * self.opt.lambda_focal
                self.loss_total += cur_loss
            elif self.criterion_names[k] == 'L2':
                cur_loss = criterion(self.full_feat, self.miss_feat) * self.opt.lambda_L2
                self.loss_total += cur_loss
            elif self.criterion_names[k] == 'ssim':
                cur_loss = -criterion(self.full_feat, self.miss_feat) * self.opt.lambda_SSIM
                self.loss_total += cur_loss
            elif self.criterion_names[k] == 'KL':
                cur_loss = criterion(normalize(self.miss_feat), normalize(self.full_feat)) * self.opt.lambda_KL
                self.loss_total += cur_loss
            setattr(self, 'loss_%s' % self.criterion_names[k], cur_loss)

        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizers[0].zero_grad()
        self.backward_G()
        self.optimizers[0].step()


#=========================================== Testing/ModDrop/ModDrop+===================================================

# import torch
# import random
# import numpy as np
# import torch.nn as nn
# from util.image_pool import ImagePool
# from .base_model import BaseModel
# from .losses import FocalLoss, dice_loss
# from . import networks
# from configurations import *
#
#
# class MsModel(BaseModel):
#     def name(self):
#         return 'MsModel'
#
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         if is_train:
#             parser.add_argument('--lambda_L2', type=int, default=2000, help='weight for L2 loss')
#             parser.add_argument('--lambda_dice', type=int, default=100, help='weight for dice loss')
#             parser.add_argument('--lambda_focal', type=int, default=10000, help='weight for focal loss')
#         return parser
#
#     def initialize(self, opt, model_suffix):
#         BaseModel.initialize(self, opt, model_suffix)
#         self.isTrain = opt.isTrain
#         self.use_dyn = opt.use_dyn
#         self.model_names = ['G']
#         self.netG = networks.define_G(opt.input_nc * len(MODALITIES), opt.init_type, opt.init_gain, self.gpu_ids, dyn=self.use_dyn)  # Network input 3 slices x 5 modalities
#         self.visual_names = ['real_mask', 'fake_mask']
#         for modality in MODALITIES:
#             self.visual_names += [modality]
#
#         if self.isTrain:
#             self.loss_names = ['total']
#             self.criterion_names = []
#             criterions = {'L2': torch.nn.MSELoss(), 'focal': FocalLoss(gamma=1, alpha=0.25).to(self.device), 'dice': dice_loss}
#             for k in criterions.keys():
#                 if k in opt.loss_to_use:
#                     self.loss_names += [k]
#                     setattr(self, 'criterion_%s' % k, criterions[k])
#                     self.criterion_names.append(k)
#             assert len(self.criterion_names), 'should use at least one loss function in L2, focal, dice'
#             self.fake_AB_pool = ImagePool(opt.pool_size)
#
#             # for feature extraction, update only the last layer, otherwise update all the parameters
#             if self.opt.feature_extract:
#                 params_to_update = []
#                 print("Params to learn:")
#                 for name, param in self.netG.named_parameters():
#                     if 'thres' in name:
#                         params_to_update.append(param)
#                     else:
#                         param.requires_grad = False
#             else:
#                 params_to_update = self.netG.parameters()
#             self.optimizers = [torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999))]
#
#     def set_input(self, input):
#         self.dc = input['dc'].to(self.device)
#         self.mc = input['mc'].to(self.device)
#
#         for modality in MODALITIES:
#             setattr(self, modality, input[modality].to(self.device))
#
#         self.real_mask = input['mask'].to(self.device)
#         self.input = torch.cat([getattr(self, k) for k in MODALITIES], 1)
#
#     def forward(self):
#         if self.use_dyn:
#             self.fake_mask = self.netG(self.input, self.mc)  # dynamic filter version
#         else:
#             self.fake_mask = self.netG(self.input)   # regular version
#
#     def backward_G(self):
#         self.loss_total = 0
#         fake_mask = (self.fake_mask + 1) / 2
#         real_mask = (self.real_mask + 1) / 2
#         for k, criterion_name in enumerate(self.criterion_names):
#             criterion = getattr(self, 'criterion_%s' % criterion_name)
#             if criterion_name == 'dice':
#                 tmp = criterion(fake_mask, real_mask) * self.opt.lambda_dice
#             elif self.criterion_names[k] == 'focal':
#                 tmp = criterion(self.fake_mask, real_mask) * self.opt.lambda_focal
#             else:
#                 tmp = criterion(self.fake_mask, self.real_mask) * self.opt.lambda_L2
#             self.loss_total += tmp
#             setattr(self, 'loss_%s' % self.criterion_names[k], tmp)
#
#         self.loss_total.backward()
#
#     def optimize_parameters(self):
#         self.forward()
#         self.optimizers[0].zero_grad()
#         self.backward_G()
#         self.optimizers[0].step()
