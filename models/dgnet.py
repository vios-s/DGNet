import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from models.meta_segmentor import *
from models.meta_styleencoder import *
from models.meta_decoder import *



class DGNet(nn.Module):
    def __init__(self, width, height, num_classes, ndf, z_length, norm, upsample, decoder_type, anatomy_out_channels, num_mask_channels):
        super(DGNet, self).__init__()
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            nclasses: number of semantice segmentation classes
        """
        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.num_mask_channels = num_mask_channels

        self.m_encoder = StyleEncoder(z_length*2)
        self.a_encoder = ContentEncoder(self.h, self.w, self.ndf, self.anatomy_out_channels, self.norm, self.upsample)
        # self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Ada_Decoder(self.decoder_type, self.anatomy_out_channels, self.z_length*2, self.num_mask_channels)

    def forward(self, x, mask, script_type, meta_loss=None, meta_step_size=0.001, stop_gradient=False, a_in=None, z_in=None):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        z_out, mu_out, logvar_out, cls_out= self.m_encoder(x, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        a_out = self.a_encoder(x, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        # seg_pred = self.segmentor(a_out, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
        #        stop_gradient=self.stop_gradient)

        z_out_tilede = None
        cls_out_tild = None

        #t0 = time.time()
        if a_in is None:
            if script_type == 'training':
                reco = self.decoder(a_out, z_out, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
                z_out_tilede, mu_out_tilde, logvar_tilde, cls_out_tild = self.m_encoder(reco, meta_loss=self.meta_loss,
                                                                                        meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
            elif script_type == 'val' or script_type == 'test':
                z_out_tilede, mu_out_tilde, logvar_tilde, cls_out_tild = self.m_encoder(x, meta_loss=self.meta_loss,
                                                                                        meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
                reco = self.decoder(a_out, z_out_tilede, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        else:
            reco = self.decoder(a_in, z_in, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)

        return reco, z_out, z_out_tilede, a_out, None, mu_out, logvar_out, cls_out, cls_out_tild
