import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops import *

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, output_dim)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        x = x.view(x.size(0), -1)

        out = linear(x, self.fc1.weight, self.fc1.bias, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        out = relu(out)
        out = linear(out, self.fc2.weight, self.fc2.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        out = relu(out)
        out = linear(out, self.fc3.weight, self.fc3.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        out = relu(out)

        return out

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class Decoder(nn.Module):
    def __init__(self, dim, output_dim=1):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain1   = AdaptiveInstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain2   = AdaptiveInstanceNorm2d(dim)

        self.conv3 = nn.Conv2d(dim, dim//2, 3, 1, 1, bias=True)
        dim //= 2
        # self.ln3 = LayerNorm(dim)
        self.bn3 = normalization(dim, norm='bn')
        self.conv4 = nn.Conv2d(dim, output_dim, 3, 1, 1, bias=True)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        # x = F.softmax(x, dim=1)

        out = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        out = self.adain1(out)
        out = conv2d(out, self.conv2.weight, self.conv2.bias, stride=1, padding=1, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        out = self.adain2(out)
        out = conv2d(out, self.conv3.weight, self.conv3.bias, stride=1, padding=1, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        # out = self.bn3(out)
        out = conv2d(out, self.conv4.weight, self.conv4.bias, stride=1, padding=1, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        out = tanh(out)
        return out

# decoder
class Ada_Decoder(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, decoder_type, anatomy_out_channels, z_length, num_mask_channels):
        super(Ada_Decoder, self).__init__()
        """
        """
        self.dec = Decoder(anatomy_out_channels)
        # MLP to generate AdaIN parameters
        self.mlp = MLP(z_length, self.get_num_adain_params(self.dec), 256, 3)

    def forward(self, a, z, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient
        # reconstruct an image
        images_recon = self.decode(a, z, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        return images_recon

    def decode(self, content, style, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient
        # decode content and style codes to an image
        adain_params = self.mlp(style, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params