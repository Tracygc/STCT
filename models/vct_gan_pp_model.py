import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms as tfs

import timm
import util.util as util
from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from .vct_gan_model import Normalize


class SobelStructure(nn.Module):
    """Extract differentiable gray, edge, and gradient responses."""

    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                 [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                 [0.0, 0.0, 0.0],
                                 [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def to_gray(self, x):
        if x.size(1) == 1:
            return x
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def normalize_positive(self, x):
        b = x.size(0)
        max_val = x.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return x / (max_val + 1e-6)

    def forward(self, x):
        gray = self.to_gray(x)
        padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(padded, self.kernel_x)
        grad_y = F.conv2d(padded, self.kernel_y)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        edge = self.normalize_positive(edge)
        low = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        return gray, low, edge, grad_x, grad_y


class ThermalStructureRecoverHead(nn.Module):
    """Small auxiliary head used only during training."""

    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class VCTGANPPModel(BaseModel):
    """VCT-GAN++: structure-aware temporal contrastive translation."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--adj_size_list', type=list, default=[2, 4, 6, 8, 12],
                            help='different scales of perception field')
        parser.add_argument('--lambda_mlp', type=float, default=1.0, help='weight of lr for discriminator')
        parser.add_argument('--lambda_motion', type=float, default=1.0, help='legacy temporal weight')
        parser.add_argument('--lambda_D_ViT', type=float, default=1.0, help='weight for ViT token discriminator')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_global', type=float, default=1.0, help='legacy global structural weight')
        parser.add_argument('--lambda_spatial', type=float, default=1.0, help='legacy local structural weight')
        parser.add_argument('--atten_layers', type=str, default='1,3,5,7,9',
                            help='compute Cross-Similarity on which ViT layers')
        parser.add_argument('--local_nums', type=int, default=256)
        parser.add_argument('--which_D_layer', type=int, default=-1)
        parser.add_argument('--side_length', type=int, default=7)

        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='include negatives from other minibatch samples for NCE')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for token NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping')

        parser.add_argument('--vit_model_name', type=str, default='vit_base_patch16_384',
                            help='training-time ViT teacher backbone')
        parser.add_argument('--vit_pretrained', type=util.str2bool, nargs='?', const=True, default=True,
                            help='load pretrained ViT teacher weights')
        parser.add_argument('--freeze_vit_teacher', type=util.str2bool, nargs='?', const=True, default=True,
                            help='freeze ViT teacher weights while keeping gradients to generated images')
        parser.add_argument('--token_prune_ratio', type=float, default=1.0,
                            help='ratio of ViT discriminator tokens kept during training')

        parser.add_argument('--lambda_structure', type=float, default=2.0,
                            help='weight for cross-modal structure consistency')
        parser.add_argument('--lambda_anti_hallucination', type=float, default=1.0,
                            help='weight for thermal-structure recoverability')
        parser.add_argument('--lambda_temporal_token', type=float, default=0.5,
                            help='weight for temporal token memory consistency')
        parser.add_argument('--lambda_temporal_luma', type=float, default=0.25,
                            help='weight for low-frequency temporal response consistency')
        parser.add_argument('--lambda_perception', type=float, default=0.5,
                            help='weight for traffic-saliency perception preservation')
        parser.add_argument('--structure_mode', type=str, default='full',
                            choices=['full', 'edge', 'edge_grad', 'contrast'],
                            help='structure ablation mode')
        parser.add_argument('--recover_mode', type=str, default='full',
                            choices=['full', 'low', 'edge'],
                            help='thermal recoverability ablation mode')
        parser.add_argument('--temporal_mode', type=str, default='full',
                            choices=['full', 'token', 'luma'],
                            help='temporal consistency ablation mode')
        parser.add_argument('--perception_mode', type=str, default='thermal_edge',
                            choices=['thermal_edge', 'thermal', 'edge'],
                            help='traffic saliency mask ablation mode')
        parser.add_argument('--use_temporal_memory_on_images', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use cross-batch token memory when only single images are available')
        parser.add_argument('--temporal_memory_momentum', type=float, default=0.9,
                            help='EMA momentum for temporal token memory')

        parser.set_defaults(pool_size=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = [
            'G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'G', 'NCE',
            'structure', 'anti_hallucination', 'temporal', 'perception'
        ]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.atten_layers = [int(i) for i in self.opt.atten_layers.split(',')]
        self.is_video_batch = False
        self.temporal_token_memory = None

        if self.isTrain:
            self.model_names = ['G', 'D_ViT', 'IRRec']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)

        if self.isTrain:
            self.netD_ViT = networks.MLPDiscriminator().to(self.device)
            self.netIRRec = ThermalStructureRecoverHead(opt.output_nc).to(self.device)
            self.netPreViT = timm.create_model(opt.vit_model_name, pretrained=opt.vit_pretrained).to(self.device)
            self.netPreViT.eval()
            if opt.freeze_vit_teacher:
                for param in self.netPreViT.parameters():
                    param.requires_grad = False

            self.structure = SobelStructure().to(self.device)
            self.resize = tfs.Resize(size=(384, 384))
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = [PatchNCELoss(opt).to(self.device) for _ in self.atten_layers]
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.l2norm = Normalize(2)

            g_params = itertools.chain(self.netG.parameters(), self.netIRRec.parameters())
            self.optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_ViT = torch.optim.Adam(self.netD_ViT.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                    betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_ViT)

    def data_dependent_initialize(self, data):
        pass

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.is_video_batch = 'A0' in input and 'A1' in input

        if self.is_video_batch:
            src0_key, src1_key = ('A0', 'A1') if AtoB else ('B0', 'B1')
            tgt0_key, tgt1_key = ('B0', 'B1') if AtoB else ('A0', 'A1')
            self.real_A0 = input[src0_key].to(self.device)
            self.real_A1 = input[src1_key].to(self.device)
            self.real_B0 = input[tgt0_key].to(self.device)
            self.real_B1 = input[tgt1_key].to(self.device)
            self.real_A = self.real_A0
            self.real_B = self.real_B0
        else:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD_ViT, True)
        self.optimizer_D_ViT.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D_ViT.step()

        self.set_requires_grad(self.netD_ViT, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.update_temporal_memory()

    def forward(self):
        if not self.opt.isTrain:
            self.fake_B = self.netG(self.real_A)
            return

        if self.is_video_batch:
            inputs = [self.real_A0, self.real_A1]
            if self.opt.nce_idt:
                inputs += [self.real_B0, self.real_B1]
            generated = self.netG(torch.cat(inputs, dim=0))
            batch = self.real_A0.size(0)
            self.fake_B0 = generated[:batch]
            self.fake_B1 = generated[batch:batch * 2]
            self.fake_B = self.fake_B0

            self.real_A_resize = self.resize(torch.cat([self.real_A0, self.real_A1], dim=0))
            self.real_B_resize = self.resize(torch.cat([self.real_B0, self.real_B1], dim=0))
            self.fake_B_resize = self.resize(torch.cat([self.fake_B0, self.fake_B1], dim=0))

            self.mutil_real_A_tokens = self.extract_teacher_tokens(self.real_A_resize, needs_input_grad=False)
            self.mutil_real_B_tokens = self.extract_teacher_tokens(self.real_B_resize, needs_input_grad=False)
            self.mutil_fake_B_tokens = self.extract_teacher_tokens(self.fake_B_resize, needs_input_grad=True)

            self.mutil_fake_B0_tokens = [tokens[:batch] for tokens in self.mutil_fake_B_tokens]
            self.mutil_fake_B1_tokens = [tokens[batch:batch * 2] for tokens in self.mutil_fake_B_tokens]

            if self.opt.nce_idt:
                self.idt_B0 = generated[batch * 2:batch * 3]
                self.idt_B1 = generated[batch * 3:batch * 4]
                idt_resize = self.resize(torch.cat([self.idt_B0, self.idt_B1], dim=0))
                self.mutil_idt_B_tokens = self.extract_teacher_tokens(idt_resize, needs_input_grad=True)
            else:
                self.mutil_idt_B_tokens = None
        else:
            inputs = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A
            generated = self.netG(inputs)
            batch = self.real_A.size(0)
            self.fake_B = generated[:batch]

            self.real_A_resize = self.resize(self.real_A)
            self.real_B_resize = self.resize(self.real_B)
            self.fake_B_resize = self.resize(self.fake_B)

            self.mutil_real_A_tokens = self.extract_teacher_tokens(self.real_A_resize, needs_input_grad=False)
            self.mutil_real_B_tokens = self.extract_teacher_tokens(self.real_B_resize, needs_input_grad=False)
            self.mutil_fake_B_tokens = self.extract_teacher_tokens(self.fake_B_resize, needs_input_grad=True)

            if self.opt.nce_idt:
                self.idt_B = generated[batch:]
                self.idt_B_resize = self.resize(self.idt_B)
                self.mutil_idt_B_tokens = self.extract_teacher_tokens(self.idt_B_resize, needs_input_grad=True)
            else:
                self.mutil_idt_B_tokens = None

    def extract_teacher_tokens(self, images, needs_input_grad):
        self.netPreViT.eval()
        if needs_input_grad:
            return self.netPreViT(images, self.atten_layers, get_tokens=True)
        with torch.no_grad():
            return self.netPreViT(images, self.atten_layers, get_tokens=True)

    def compute_D_loss(self):
        fake_B_tokens = self.mutil_fake_B_tokens[self.opt.which_D_layer].detach()
        real_B_tokens = self.mutil_real_B_tokens[self.opt.which_D_layer]

        fake_B_tokens = self.prune_tokens(self.cat_results(fake_B_tokens, self.opt.adj_size_list))
        real_B_tokens = self.prune_tokens(self.cat_results(real_B_tokens, self.opt.adj_size_list))

        pred_fake_ViT = self.netD_ViT(fake_B_tokens)
        self.loss_D_fake_ViT = self.criterionGAN(pred_fake_ViT, False).mean() * self.opt.lambda_D_ViT

        pred_real_ViT = self.netD_ViT(real_B_tokens)
        self.loss_D_real_ViT = self.criterionGAN(pred_real_ViT, True).mean() * self.opt.lambda_D_ViT

        self.loss_D_ViT = (self.loss_D_fake_ViT + self.loss_D_real_ViT) * 0.5
        return self.loss_D_ViT

    def compute_G_loss(self):
        zero = self.zero_loss()

        if self.opt.lambda_GAN > 0.0:
            fake_B_tokens = self.mutil_fake_B_tokens[self.opt.which_D_layer]
            fake_B_tokens = self.prune_tokens(self.cat_results(fake_B_tokens, self.opt.adj_size_list))
            pred_fake_ViT = self.netD_ViT(fake_B_tokens)
            self.loss_G_GAN_ViT = self.criterionGAN(pred_fake_ViT, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ViT = zero

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.mutil_real_A_tokens, self.mutil_fake_B_tokens)
        else:
            self.loss_NCE = zero

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0 and self.mutil_idt_B_tokens is not None:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.mutil_real_B_tokens, self.mutil_idt_B_tokens)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_structure = self.calculate_structure_loss() if self.opt.lambda_structure > 0.0 else zero
        self.loss_anti_hallucination = (
            self.calculate_anti_hallucination_loss() if self.opt.lambda_anti_hallucination > 0.0 else zero
        )
        self.loss_temporal = self.calculate_temporal_loss() if self.temporal_loss_enabled() else zero
        self.loss_perception = self.calculate_perception_loss() if self.opt.lambda_perception > 0.0 else zero

        self.loss_G = (
            self.loss_G_GAN_ViT + loss_NCE_both + self.loss_structure +
            self.loss_anti_hallucination + self.loss_temporal + self.loss_perception
        )
        return self.loss_G

    def zero_loss(self):
        return self.fake_B.sum() * 0.0

    def temporal_loss_enabled(self):
        if self.opt.lambda_temporal_token <= 0.0 and self.opt.lambda_temporal_luma <= 0.0:
            return False
        return self.is_video_batch or (
            self.opt.use_temporal_memory_on_images and self.temporal_token_memory is not None
        )

    def calculate_structure_loss(self):
        src, fake = self.structure_pairs()
        _, _, src_edge, src_gx, src_gy = self.structure(src)
        _, _, fake_edge, fake_gx, fake_gy = self.structure(fake)

        edge_loss = self.criterionL1(fake_edge, src_edge)
        grad_loss = 0.5 * (self.criterionL1(fake_gx, src_gx) + self.criterionL1(fake_gy, src_gy))
        contrast_loss = self.structure_contrastive_loss(src_edge, fake_edge)
        if self.opt.structure_mode == 'edge':
            loss = edge_loss
        elif self.opt.structure_mode == 'edge_grad':
            loss = edge_loss + 0.25 * grad_loss
        elif self.opt.structure_mode == 'contrast':
            loss = contrast_loss
        else:
            loss = edge_loss + 0.25 * grad_loss + 0.5 * contrast_loss
        return loss * self.opt.lambda_structure

    def calculate_anti_hallucination_loss(self):
        src, fake = self.structure_pairs()
        _, src_low, src_edge, _, _ = self.structure(src)
        rec = self.netIRRec(fake)
        rec_low = torch.tanh(rec[:, 0:1])
        rec_edge = torch.sigmoid(rec[:, 1:2])
        low_loss = self.criterionL1(rec_low, src_low)
        edge_loss = self.criterionL1(rec_edge, src_edge.detach())
        if self.opt.recover_mode == 'low':
            loss = low_loss
        elif self.opt.recover_mode == 'edge':
            loss = edge_loss
        else:
            loss = low_loss + edge_loss
        return loss * self.opt.lambda_anti_hallucination

    def calculate_temporal_loss(self):
        zero = self.zero_loss()
        token_loss = zero
        luma_loss = zero

        if self.is_video_batch:
            token_loss = self.temporal_token_loss(self.mutil_fake_B0_tokens, self.mutil_fake_B1_tokens)
            luma_loss = self.temporal_response_loss(self.real_A0, self.real_A1, self.fake_B0, self.fake_B1)
        elif self.opt.use_temporal_memory_on_images and self.temporal_token_memory is not None:
            token_loss = self.temporal_token_loss(self.temporal_token_memory, self.mutil_fake_B_tokens)

        if self.opt.temporal_mode == 'token':
            return token_loss * self.opt.lambda_temporal_token
        if self.opt.temporal_mode == 'luma':
            return luma_loss * self.opt.lambda_temporal_luma
        return token_loss * self.opt.lambda_temporal_token + luma_loss * self.opt.lambda_temporal_luma

    def calculate_perception_loss(self):
        src, fake = self.structure_pairs()
        src_gray, src_low, src_edge, _, _ = self.structure(src)
        fake_gray, fake_low, fake_edge, _, _ = self.structure(fake)
        hot = (src_gray.detach() + 1.0) * 0.5
        if self.opt.perception_mode == 'thermal':
            mask = hot
        elif self.opt.perception_mode == 'edge':
            mask = src_edge.detach()
        else:
            mask = (0.6 * hot + 0.4 * src_edge.detach()).clamp(0.0, 1.0)
        mask = mask / (mask.mean(dim=(2, 3), keepdim=True) + 1e-6)
        edge_loss = self.masked_l1(fake_edge, src_edge.detach(), mask)
        low_loss = self.masked_l1(fake_low, src_low.detach(), mask)
        return (edge_loss + 0.25 * low_loss) * self.opt.lambda_perception

    def structure_pairs(self):
        if self.is_video_batch:
            src = torch.cat([self.real_A0, self.real_A1], dim=0)
            fake = torch.cat([self.fake_B0, self.fake_B1], dim=0)
        else:
            src = self.real_A
            fake = self.fake_B
        return src, fake

    def structure_contrastive_loss(self, src_edge, fake_edge):
        src_pool = F.adaptive_avg_pool2d(src_edge, output_size=(16, 16)).flatten(1)
        fake_pool = F.adaptive_avg_pool2d(fake_edge, output_size=(16, 16)).flatten(1)
        return (1.0 - F.cosine_similarity(fake_pool, src_pool.detach(), dim=1)).mean()

    def temporal_token_loss(self, previous_tokens, current_tokens):
        total = self.zero_loss()
        count = 0
        for prev, cur in zip(previous_tokens, current_tokens):
            prev_flat = prev.detach().reshape(prev.size(0), -1)
            cur_flat = cur.reshape(cur.size(0), -1)
            if prev_flat.shape != cur_flat.shape:
                continue
            total = total + (1.0 - F.cosine_similarity(cur_flat, prev_flat, dim=1)).mean()
            count += 1
        if count == 0:
            return self.zero_loss()
        return total / count

    def temporal_response_loss(self, src_prev, src_cur, fake_prev, fake_cur):
        src_prev_gray, src_prev_low, src_prev_edge, _, _ = self.structure(src_prev)
        src_cur_gray, src_cur_low, src_cur_edge, _, _ = self.structure(src_cur)
        fake_prev_gray, fake_prev_low, fake_prev_edge, _, _ = self.structure(fake_prev)
        fake_cur_gray, fake_cur_low, fake_cur_edge, _, _ = self.structure(fake_cur)

        src_delta = (src_cur_low - src_prev_low).detach()
        fake_delta = fake_cur_low - fake_prev_low.detach()
        low_loss = self.criterionL1(fake_delta, src_delta)

        src_edge_delta = (src_cur_edge - src_prev_edge).detach()
        fake_edge_delta = fake_cur_edge - fake_prev_edge.detach()
        edge_loss = self.criterionL1(fake_edge_delta, src_edge_delta)
        return low_loss + 0.5 * edge_loss

    def masked_l1(self, pred, target, mask):
        return (torch.abs(pred - target) * mask).mean()

    def update_temporal_memory(self):
        if not self.opt.use_temporal_memory_on_images:
            return
        current = [token.detach() for token in self.mutil_fake_B_tokens]
        if self.temporal_token_memory is None:
            self.temporal_token_memory = current
            return
        momentum = self.opt.temporal_memory_momentum
        updated = []
        for old, new in zip(self.temporal_token_memory, current):
            if old.shape == new.shape:
                updated.append(old * momentum + new * (1.0 - momentum))
            else:
                updated.append(new)
        self.temporal_token_memory = updated

    def prune_tokens(self, tokens):
        ratio = max(0.0, min(1.0, self.opt.token_prune_ratio))
        if ratio >= 1.0 or tokens.size(1) <= 1:
            return tokens
        keep = max(1, int(tokens.size(1) * ratio))
        token_ids = torch.randperm(tokens.size(1), device=tokens.device)[:keep]
        return tokens[:, token_ids, :]

    def tokens_concat(self, origin_tokens, adjacent_size):
        adj_size = adjacent_size
        batch, token_num, channels = origin_tokens.shape[0], origin_tokens.shape[1], origin_tokens.shape[2]
        side = int(math.sqrt(token_num))
        if side * side != token_num:
            print('Error! Not a square!')
        token_map = origin_tokens.clone().reshape(batch, side, side, channels)
        cut_patch_list = []
        for i in range(0, side, adj_size):
            for j in range(0, side, adj_size):
                i_left = i
                i_right = i + adj_size + 1 if i + adj_size <= side else side + 1
                j_left = j
                j_right = j + adj_size if j + adj_size <= side else side + 1

                cut_patch = token_map[:, i_left:i_right, j_left:j_right, :]
                cut_patch = cut_patch.reshape(batch, -1, channels)
                cut_patch = torch.mean(cut_patch, dim=1, keepdim=True)
                cut_patch_list.append(cut_patch)

        return torch.cat(cut_patch_list, dim=1)

    def cat_results(self, origin_tokens, adj_size_list):
        res_list = [origin_tokens]
        for adj_size in adj_size_list:
            res_list.append(self.tokens_concat(origin_tokens, adj_size))
        return torch.cat(res_list, dim=1)

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.atten_layers)
        src_pool, sample_ids = self.downsample(src, self.opt.num_patches, None)
        tgt_pool, _ = self.downsample(tgt, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(tgt_pool, src_pool, self.criterionNCE):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def downsample(self, tokens, num_patches=256, patch_ids=None):
        return_ids = []
        return_tokens = []

        for token_id, token in enumerate(tokens):
            token_reshape = token.flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[token_id]
                else:
                    patch_id = np.random.permutation(token_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches * num_patches, patch_id.shape[0]))]
                if isinstance(patch_id, torch.Tensor):
                    patch_id = patch_id.to(device=token.device, dtype=torch.long)
                else:
                    patch_id = torch.tensor(patch_id, dtype=torch.long, device=token.device)
                token_sample = token_reshape[:, patch_id]
                token_sample = token_sample.view(num_patches, -1)
            else:
                token_sample = token_reshape
                patch_id = []
            return_ids.append(patch_id)
            token_sample = self.l2norm(token_sample)
            return_tokens.append(token_sample)
        return return_tokens, return_ids
