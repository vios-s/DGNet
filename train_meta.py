import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from eval import eval_dgnet
from tqdm import tqdm
import logging
from metrics.focal_loss import FocalLoss
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn.functional as F
import utils
from loaders.mms_dataloader_meta_split import get_meta_split_data_loaders
import models
import losses
from torch.utils.tensorboard import SummaryWriter
import time


def get_args():
    usage_text = (
        "DGNet Pytorch Implementation"
        "Usage:  python train_meta.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type=int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')
    parser.add_argument('-tc', '--tcp', type=str, default='temp_checkpoints/', help='The name of the checkpoints.')
    parser.add_argument('-t', '--tv', type=str, default='D', help='The name of the target vendor.')
    parser.add_argument('-w', '--wc', type=str, default='DGNet_LR00002_LDv5', help='The name of the writter summary.')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='dgnet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.00004', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    parser.add_argument('--decoder_type', type=str, default='film', help='Choose decoder type between FiLM and SPADE')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')

    return parser.parse_args()

# python train_meta.py -e 80 -c cp_dgnet_meta_100_tvA/ -t A -w DGNetRE_COM_META_100_tvA -g 0
# python train_meta.py -e 80 -c cp_dgnet_meta_100_tvB/ -t B -w DGNetRE_COM_META_100_tvB -g 1
# python train_meta.py -e 80 -c cp_dgnet_meta_100_tvC/ -t C -w DGNetRE_COM_META_100_tvC -g 2
# python train_meta.py -e 80 -c cp_dgnet_meta_100_tvD/ -t D -w DGNetRE_COM_META_100_tvD -g 3
# k_un = 1
# k1 = 20
# k2 = 2
# opt_patience = 4

# python train_meta.py -e 100 -c cp_dgnet_meta_50_tvA/ -t A -w DGNetRE_COM_META_50_tvA -g 0
# python train_meta.py -e 100 -c cp_dgnet_meta_50_tvB/ -t B -w DGNetRE_COM_META_50_tvB -g 1
# python train_meta.py -e 100 -c cp_dgnet_meta_50_tvC/ -t C -w DGNetRE_COM_META_50_tvC -g 2
# python train_meta.py -e 100 -c cp_dgnet_meta_50_tvD/ -t D -w DGNetRE_COM_META_50_tvD -g 3
# k_un = 1
# k1 = 20
# k2 = 3
# opt_patience = 4

# python train_meta.py -e 120 -c cp_dgnet_meta_20_tvA/ -t A -w DGNetRE_COM_META_20_tvA -g 0
# python train_meta.py -e 120 -c cp_dgnet_meta_20_tvB/ -t B -w DGNetRE_COM_META_20_tvB -g 1
# python train_meta.py -e 120 -c cp_dgnet_meta_20_tvC/ -t C -w DGNetRE_COM_META_20_tvC -g 2
# python train_meta.py -e 120 -c cp_dgnet_meta_20_tvD/ -t D -w DGNetRE_COM_META_20_tvD -g 3
# k_un = 1
# k1 = 30
# k2 = 3
# opt_patience = 4

# python train_meta.py -e 150 -c cp_dgnet_meta_5_tvA/ -t A -w DGNetRE_COM_META_5_tvA -g 0
# python train_meta.py -e 150 -c cp_dgnet_meta_5_tvB/ -t B -w DGNetRE_COM_META_5_tvB -g 1
# python train_meta.py -e 150 -c cp_dgnet_meta_5_tvC/ -t C -w DGNetRE_COM_META_5_tvC -g 2
# python train_meta.py -e 150 -c cp_dgnet_meta_5_tvD/ -t D -w DGNetRE_COM_META_5_tvD -g 3
k_un = 1
k1 = 30
k2 = 3
opt_patience = 4

def latent_norm(a):
    n_batch, n_channel, _, _ = a.size()
    for batch in range(n_batch):
        for channel in range(n_channel):
            a_min = a[batch,channel,:,:].min()
            a_max = a[batch, channel, :, :].max()
            a[batch,channel,:,:] += a_min
            a[batch, channel, :, :] /= a_max - a_min
    return a

def train_net(args):
    best_dice = 0
    best_lv = 0
    best_myo = 0
    best_rv = 0

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    dir_checkpoint = args.cp
    test_vendor = args.tv
    wc = args.wc

    #Model selection and initialization
    model_params = {
        'width': 288,
        'height': 288,
        'ndf': 64,
        'norm': "batchnorm",
        'upsample': "nearest",
        'num_classes': 3,
        'decoder_type': args.decoder_type,
        'anatomy_out_channels': 8,
        'z_length': 8,
        'num_mask_channels': 8,

    }
    model = models.get_model(args.model_name, model_params)
    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)

    # size:
    # X: N, 1, 224, 224
    # Y: N, 3, 224, 224

    _, domain_1_unlabeled_loader, \
    _, domain_2_unlabeled_loader,\
    _, domain_3_unlabeled_loader, \
    test_loader, \
    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset = get_meta_split_data_loaders(batch_size//2, test_vendor=test_vendor, image_size=224)


    val_dataset = ConcatDataset([domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=2)
    n_val = len(val_dataset)
    print(n_val)
    print(len(test_loader))
    print(len(domain_1_unlabeled_loader))
    print(len(domain_2_unlabeled_loader))
    print(len(domain_3_unlabeled_loader))

    d_len = []
    d_len.append(len(domain_1_labeled_dataset))
    d_len.append(len(domain_2_labeled_dataset))
    d_len.append(len(domain_3_labeled_dataset))
    long_len = d_len[0]
    for i in range(len(d_len)):
        long_len = d_len[i] if d_len[i]>=long_len else long_len
    print(long_len)

    new_d_1 = domain_1_labeled_dataset
    for i in range(long_len//d_len[0]+1):
        if long_len == d_len[0]:
            break
        new_d_1 = ConcatDataset([new_d_1, domain_1_labeled_dataset])
    domain_1_labeled_dataset = new_d_1
    domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_size//2, shuffle=False,
                                        drop_last=True, num_workers=2, pin_memory=True)

    new_d_2 = domain_2_labeled_dataset
    for i in range(long_len//d_len[1]+1):
        if long_len == d_len[1]:
            break
        new_d_2 = ConcatDataset([new_d_2, domain_2_labeled_dataset])
    domain_2_labeled_dataset = new_d_2
    domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_size//2, shuffle=False,
                                        drop_last=True, num_workers=2, pin_memory=True)

    new_d_3 = domain_3_labeled_dataset
    for i in range(long_len//d_len[2]+1):
        if long_len == d_len[2]:
            break
        new_d_3 = ConcatDataset([new_d_3, domain_3_labeled_dataset])
    domain_3_labeled_dataset = new_d_3
    domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_size//2, shuffle=False,
                                        drop_last=True, num_workers=2, pin_memory=True)

    print(len(domain_1_labeled_loader))
    print(len(domain_2_labeled_loader))
    print(len(domain_3_labeled_loader))

    #metrics initialization
    # l2_distance = nn.MSELoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    l1_distance = nn.L1Loss().to(device)
    focal = FocalLoss()

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=opt_patience)

    writer = SummaryWriter(comment=wc)

    global_step = 0
    for epoch in range(epochs):
        model.train()
        with tqdm(total=long_len, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            domain_1_labeled_itr = iter(domain_1_labeled_loader)
            domain_2_labeled_itr = iter(domain_2_labeled_loader)
            domain_3_labeled_itr = iter(domain_3_labeled_loader)
            domain_labeled_iter_list = [domain_1_labeled_itr, domain_2_labeled_itr, domain_3_labeled_itr]

            domain_1_unlabeled_itr = iter(domain_1_unlabeled_loader)
            domain_2_unlabeled_itr = iter(domain_2_unlabeled_loader)
            domain_3_unlabeled_itr = iter(domain_3_unlabeled_loader)
            domain_unlabeled_iter_list = [domain_1_unlabeled_itr, domain_2_unlabeled_itr, domain_3_unlabeled_itr]


            for num_itr in range(long_len//batch_size):
                # Randomly choosing meta train and meta test domains
                domain_list = np.random.permutation(3)
                meta_train_domain_list = domain_list[:2]
                meta_test_domain_list = domain_list[2]

                meta_train_imgs = []
                meta_train_masks = []
                meta_train_labels = []
                meta_test_imgs = []
                meta_test_masks = []
                meta_test_labels = []
                meta_test_un_imgs = []
                meta_test_un_labels = []

                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_train_domain_list[0]])
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)

                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_train_domain_list[1]])
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)

                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_test_domain_list])
                meta_test_imgs.append(imgs)
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)
                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_test_domain_list])
                meta_test_imgs.append(imgs)
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)

                imgs, labels = next(domain_unlabeled_iter_list[meta_test_domain_list])
                meta_test_un_imgs.append(imgs)
                meta_test_un_labels.append(labels)
                imgs, labels = next(domain_unlabeled_iter_list[meta_test_domain_list])
                meta_test_un_imgs.append(imgs)
                meta_test_un_labels.append(labels)

                meta_train_imgs = torch.cat((meta_train_imgs[0], meta_train_imgs[1]), dim=0)
                meta_train_masks = torch.cat((meta_train_masks[0], meta_train_masks[1]), dim=0)
                meta_train_labels = torch.cat((meta_train_labels[0], meta_train_labels[1]), dim=0)
                meta_test_imgs = torch.cat((meta_test_imgs[0], meta_test_imgs[1]), dim=0)
                meta_test_masks = torch.cat((meta_test_masks[0], meta_test_masks[1]), dim=0)
                meta_test_labels = torch.cat((meta_test_labels[0], meta_test_labels[1]), dim=0)
                meta_test_un_imgs = torch.cat((meta_test_un_imgs[0], meta_test_un_imgs[1]), dim=0)
                meta_test_un_labels = torch.cat((meta_test_un_labels[0], meta_test_un_labels[1]), dim=0)

                meta_train_un_imgs = []
                meta_train_un_labels = []
                for i in range(k_un):
                    train_un_imgs = []
                    train_un_labels = []
                    un_imgs, un_labels = next(domain_unlabeled_iter_list[meta_train_domain_list[0]])
                    train_un_imgs.append(un_imgs)
                    train_un_labels.append(un_labels)
                    un_imgs, un_labels = next(domain_unlabeled_iter_list[meta_train_domain_list[1]])
                    train_un_imgs.append(un_imgs)
                    train_un_labels.append(un_labels)
                    meta_train_un_imgs.append(torch.cat((train_un_imgs[0], train_un_imgs[1]), dim=0))
                    meta_train_un_labels.append(torch.cat((train_un_labels[0], train_un_labels[1]), dim=0))

                total_meta_un_loss = 0.0
                for i in range(k_un):
                    # meta-train: 1. load meta-train data 2. calculate meta-train loss
                    ###############################Meta train#######################################################
                    un_imgs = meta_train_un_imgs[i].to(device=device, dtype=torch.float32)
                    un_labels = meta_train_un_labels[i].to(device=device, dtype=torch.float32)

                    un_reco, un_z_out, un_z_tilde, un_a_out, _, un_mu, un_logvar, un_cls_out, _ = model(un_imgs, true_masks, 'training')

                    un_a_feature = F.softmax(un_a_out, dim=1)
                    # un_a_feature = un_a_feature[:,4:,:,:]
                    # un_seg_pred = un_a_out[:,:4,:,:]

                    latent_dim = un_a_feature.size(1)
                    un_a_feature = un_a_out.permute(0, 2, 3, 1).contiguous().view(-1, latent_dim)
                    un_a_feature = un_a_feature[torch.randperm(len(un_a_feature))]
                    un_U_a, un_S_a, un_V_a = torch.svd(un_a_feature[0:2000])

                    # loss_low_rank_Un_a = 0.1*torch.sum(un_S_a)
                    loss_low_rank_Un_a = un_S_a[4]

                    un_reco_loss = l1_distance(un_reco, un_imgs)
                    un_regression_loss = l1_distance(un_z_tilde, un_z_out)

                    kl_loss1 = losses.KL_divergence(un_logvar[:, :8], un_mu[:, :8])
                    kl_loss2 = losses.KL_divergence(un_logvar[:, 8:], un_mu[:, 8:])
                    hsic_loss = losses.HSIC_lossfunc(un_z_out[:, :8], un_z_out[:, 8:])
                    un_kl_loss = kl_loss1 + kl_loss2 + hsic_loss

                    d_cls = criterion(un_cls_out, un_labels)
                    un_batch_loss = un_reco_loss + (0.1*un_regression_loss) + 0.1*un_kl_loss + d_cls + 0.1*loss_low_rank_Un_a

                    total_meta_un_loss += un_batch_loss

                    # meta-test: 1. load meta-test data 2. calculate meta-test loss
                    ###############################Meta test#######################################################
                    un_imgs = meta_test_un_imgs.to(device=device, dtype=torch.float32)
                    un_labels = meta_test_un_labels.to(device=device, dtype=torch.float32)
                    un_reco, un_z_out, un_z_tilde, un_a_out, _, un_mu, un_logvar, un_cls_out, _ = model(
                        un_imgs, true_masks, 'training', meta_loss=un_batch_loss)

                    un_seg_pred = un_a_out[:, :4, :, :]
                    sf_un_seg_pred = F.softmax(un_seg_pred, dim=1)

                    un_reco_loss = l1_distance(un_reco, un_imgs)
                    un_regression_loss = l1_distance(un_z_tilde, un_z_out)

                    # kl_loss1 = losses.KL_divergence(un_logvar[:, :8], un_mu[:, :8])
                    # kl_loss2 = losses.KL_divergence(un_logvar[:, 8:], un_mu[:, 8:])
                    # hsic_loss = losses.HSIC_lossfunc(un_z_out[:, :8], un_z_out[:, 8:])
                    # un_kl_loss = kl_loss1 + kl_loss2 + hsic_loss

                    d_cls = criterion(un_cls_out, un_labels)
                    un_batch_loss = un_reco_loss + d_cls

                    total_meta_un_loss += un_batch_loss

                    writer.add_scalar('Meta_test_loss/un_reco_loss', un_reco_loss.item(), global_step)
                    writer.add_scalar('Meta_test_loss/un_regression_loss', un_regression_loss.item(), global_step)
                    # writer.add_scalar('Meta_test_loss/un_kl_loss', un_kl_loss.item(), global_step)
                    writer.add_scalar('Meta_test_loss/d_cls', d_cls.item(), global_step)
                    # writer.add_scalar('Meta_test_loss/loss_low_rank_Un_a', loss_low_rank_Un_a.item(), un_step)
                    writer.add_scalar('Meta_test_loss/un_batch_loss', un_batch_loss.item(), global_step)

                    optimizer.zero_grad()
                    total_meta_un_loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()

                total_meta_loss = 0.0
                # meta-train: 1. load meta-train data 2. calculate meta-train loss
                ###############################Meta train#######################################################
                imgs = meta_train_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = meta_train_masks.clone().to(device=device, dtype=torch.long)
                true_masks = meta_train_masks.to(device=device, dtype=mask_type)
                labels = meta_train_labels.to(device=device, dtype=torch.float32)

                reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = model(imgs, true_masks, 'training')

                # mode-1 flattering and change the original 4,8,224,224 features to 4x224x224, 8
                # randomly pick 4000, 8 features to calculate the singular values
                a_feature = F.softmax(a_out, dim=1)
                # a_feature = a_feature[:, 4:, :, :]
                seg_pred = a_out[:, :4, :, :]

                latent_dim = a_feature.size(1)
                a_feature = a_feature.permute(0, 2, 3, 1).contiguous().view(-1, latent_dim)
                a_feature = a_feature[torch.randperm(len(a_feature))]
                U_a, S_a, V_a = torch.svd(a_feature[0:2000])

                # loss_low_rank_a = 0.1*torch.sum(S_a)
                loss_low_rank_a = S_a[4]

                reco_loss = l1_distance(reco, imgs)
                kl_loss1 = losses.KL_divergence(logvar[:,:8], mu[:,:8])
                kl_loss2 = losses.KL_divergence(logvar[:,8:], mu[:,8:])
                hsic_loss = losses.HSIC_lossfunc(z_out[:,:8], z_out[:,8:])
                kl_loss = kl_loss1 + kl_loss2 + hsic_loss
                regression_loss = l1_distance(z_out_tilde, z_out)

                sf_seg = F.softmax(seg_pred, dim=1)
                dice_loss_lv = losses.dice_loss(sf_seg[:,0,:,:], true_masks[:,0,:,:])
                dice_loss_myo = losses.dice_loss(sf_seg[:,1,:,:], true_masks[:,1,:,:])
                dice_loss_rv = losses.dice_loss(sf_seg[:,2,:,:], true_masks[:,2,:,:])
                dice_loss_bg = losses.dice_loss(sf_seg[:, 3, :, :], true_masks[:, 3, :, :])
                loss_dice = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

                ce_target = ce_mask[:, 3, :, :]*0 + ce_mask[:, 0, :, :]*1 + ce_mask[:, 1, :, :]*2 + ce_mask[:, 2, :, :]*3

                seg_pred_swap = torch.cat((seg_pred[:,3,:,:].unsqueeze(1), seg_pred[:,:3,:,:]), dim=1)

                loss_focal = focal(seg_pred_swap, ce_target)

                d_cls = criterion(cls_out, labels)
                d_losses = d_cls

                batch_loss = reco_loss + (0.1 * regression_loss) + 0.1*kl_loss + 5*loss_dice + 5*loss_focal + d_losses + 0.1*loss_low_rank_a

                total_meta_loss += batch_loss

                writer.add_scalar('Meta_train_Loss/loss_dice', loss_dice.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_lv', dice_loss_lv.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_myo', dice_loss_myo.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_rv', dice_loss_rv.item(), global_step)
                writer.add_scalar('Meta_train_Loss/loss_focal', loss_focal.item(), global_step)
                writer.add_scalar('Meta_train_Loss/kl_loss', kl_loss.item(), global_step)
                writer.add_scalar('Meta_train_Loss/loss_low_rank_a', loss_low_rank_a.item(), global_step)
                writer.add_scalar('Meta_train_Loss/batch_loss', batch_loss.item(), global_step)

                # meta-test: 1. load meta-test data 2. calculate meta-test loss
                ###############################Meta test#######################################################
                imgs = meta_test_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = meta_test_masks.clone().to(device=device, dtype=torch.long)
                true_masks = meta_test_masks.to(device=device, dtype=mask_type)
                labels = meta_test_labels.to(device=device, dtype=torch.float32)
                reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = model(imgs, true_masks, 'training', meta_loss=batch_loss)

                # mode-1 flattering and change the original 4,8,224,224 features to 4x224x224, 8
                # randomly pick 4000, 8 features to calculate the singular values
                # latent_dim = a_out.size(1)
                # a_feature = a_out.permute(0, 2, 3, 1).contiguous().view(-1, latent_dim)
                # a_feature = a_feature[torch.randperm(len(a_feature))]
                # U_a, S_a, V_a = torch.svd(a_feature[0:2000])

                seg_pred = a_out[:, :4, :, :]

                reco_loss = l1_distance(reco, imgs)
                # kl_loss = losses.KL_divergence(logvar, mu)
                # regression_loss = l1_distance(z_out_tilde, z_out)

                sf_seg = F.softmax(seg_pred, dim=1)
                dice_loss_lv = losses.dice_loss(sf_seg[:,0,:,:], true_masks[:,0,:,:])
                dice_loss_myo = losses.dice_loss(sf_seg[:,1,:,:], true_masks[:,1,:,:])
                dice_loss_rv = losses.dice_loss(sf_seg[:,2,:,:], true_masks[:,2,:,:])
                dice_loss_bg = losses.dice_loss(sf_seg[:, 3, :, :], true_masks[:, 3, :, :])
                loss_dice = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

                ce_target = ce_mask[:, 3, :, :]*0 + ce_mask[:, 0, :, :]*1 + ce_mask[:, 1, :, :]*2 + ce_mask[:, 2, :, :]*3

                seg_pred_swap = torch.cat((seg_pred[:,3,:,:].unsqueeze(1), seg_pred[:,:3,:,:]), dim=1)

                loss_focal = focal(seg_pred_swap, ce_target)

                d_cls = criterion(cls_out, labels)
                d_losses = d_cls

                batch_loss = 5*loss_dice + 5*loss_focal + reco_loss + d_losses
                total_meta_loss += batch_loss

                optimizer.zero_grad()
                total_meta_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': total_meta_loss.item()})
                pbar.update(imgs.shape[0])

                if (epoch + 1) > (k1) and (epoch + 1) % k2 == 0:
                    if global_step % ((long_len//batch_size) // 2) == 0:
                        a_feature = F.softmax(a_out, dim=1)
                        a_feature = latent_norm(a_feature)
                        writer.add_images('Meta_train_images/train', imgs, global_step)
                        writer.add_images('Meta_train_images/a_out0', a_feature[:,0,:,:].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out1', a_feature[:, 1, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out2', a_feature[:, 2, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out3', a_feature[:, 3, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out4', a_feature[:, 4, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out5', a_feature[:, 5, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out6', a_feature[:, 6, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/a_out7', a_feature[:, 7, :, :].unsqueeze(1), global_step)
                        writer.add_images('Meta_train_images/train_reco', reco, global_step)
                        writer.add_images('Meta_train_images/train_true', true_masks[:,0:3,:,:], global_step)
                        writer.add_images('Meta_train_images/train_pred', sf_seg[:,0:3,:,:] > 0.5, global_step)
                        writer.add_images('Meta_test_images/train_un_img', un_imgs, global_step)
                        writer.add_images('Meta_test_images/train_un_mask', sf_un_seg_pred[:, 0:3, :, :] > 0.5, global_step)

                global_step += 1

            if optimizer.param_groups[0]['lr']<=2e-8:
                print('Converge')
            if (epoch + 1) > k1 and (epoch + 1) % k2 == 0:

                val_score, val_lv, val_myo, val_rv = eval_dgnet(model, val_loader, device, mode='val')
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                logging.info('Validation Dice Coeff: {}'.format(val_score))
                logging.info('Validation LV Dice Coeff: {}'.format(val_lv))
                logging.info('Validation MYO Dice Coeff: {}'.format(val_myo))
                logging.info('Validation RV Dice Coeff: {}'.format(val_rv))

                writer.add_scalar('Dice/val', val_score, epoch)
                writer.add_scalar('Dice/val_lv', val_lv, epoch)
                writer.add_scalar('Dice/val_myo', val_myo, epoch)
                writer.add_scalar('Dice/val_rv', val_rv, epoch)

                initial_itr = 0
                for imgs, true_masks in test_loader:
                    if initial_itr == 5:
                        model.eval()
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        with torch.no_grad():
                            reco, z_out, z_out_tilde, a_out, seg_pred, mu, logvar, _, _ = model(imgs, true_masks,
                                                                                                'test')
                        seg_pred = a_out[:, :4, :, :]
                        mask_type = torch.float32
                        true_masks = true_masks.to(device=device, dtype=mask_type)
                        sf_seg_pred = F.softmax(seg_pred, dim=1)
                        writer.add_images('Test_images/test', imgs, epoch)
                        writer.add_images('Test_images/test_reco', reco, epoch)
                        writer.add_images('Test_images/test_true', true_masks[:, 0:3, :, :], epoch)
                        writer.add_images('Test_images/test_pred', sf_seg_pred[:, 0:3, :, :] > 0.5, epoch)
                        model.train()
                        break
                    else:
                        pass
                    initial_itr += 1
                test_score, test_lv, test_myo, test_rv = eval_dgnet(model, test_loader, device, mode='test')

                if best_dice < test_score:
                    best_dice = test_score
                    best_lv = test_lv
                    best_myo = test_myo
                    best_rv = test_rv
                    print("Epoch checkpoint")
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(model.state_dict(),
                               dir_checkpoint + 'CP_epoch.pth')
                    logging.info('Checkpoint saved !')
                else:
                    pass
                logging.info('Best Dice Coeff: {}'.format(best_dice))
                logging.info('Best LV Dice Coeff: {}'.format(best_lv))
                logging.info('Best MYO Dice Coeff: {}'.format(best_myo))
                logging.info('Best RV Dice Coeff: {}'.format(best_rv))
                writer.add_scalar('Dice/test', test_score, epoch)
                writer.add_scalar('Dice/test_lv', test_lv, epoch)
                writer.add_scalar('Dice/test_myo', test_myo, epoch)
                writer.add_scalar('Dice/test_rv', test_rv, epoch)
        writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)

    train_net(args)
