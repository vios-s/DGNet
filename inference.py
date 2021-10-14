import torch
import argparse
import torch.nn.functional as F
import statistics
import utils
from loaders.mms_dataloader_meta_split_test import get_meta_split_data_loaders
import models
from metrics.dice_loss import dice_coeff
from metrics.hausdorff import hausdorff_distance

# python inference.py -bs 1 -c cp_dgnet_gan_meta_dir_5_tvA/ -t A -mn dgnet -g 0


def get_args():
    usage_text = (
        "SNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-bs','--batch_size', type=int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')
    parser.add_argument('-t', '--tv', type=str, default='D', help='The name of the target vendor.')
    parser.add_argument('-w', '--wc', type=str, default='DGNet_LR00002_LDv5', help='The name of the writter summary.')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='dgnet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.00002', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    parser.add_argument('--decoder_type', type=str, default='film', help='Choose decoder type between FiLM and SPADE')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')

    return parser.parse_args()

args = get_args()
device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

batch_size = args.batch_size
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

dir_checkpoint = args.cp
test_vendor = args.tv
# wc = args.wc
model_name = args.model_name

# Model selection and initialization
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
model = models.get_model(model_name, model_params)
num_params = utils.count_parameters(model)
print('Model Parameters: ', num_params)
model.load_state_dict(torch.load(dir_checkpoint+'CP_epoch.pth', map_location=device))
model.to(device)

# writer = SummaryWriter(comment=wc)

_, _, \
_, _, \
_, _, \
test_loader, \
_, _, _ = get_meta_split_data_loaders(
    batch_size, test_vendor=test_vendor, image_size=224)

step = 0
tot = []
tot_sub = []
tot_hsd = []
tot_sub_hsd = []
flag = '000'
# i = 0
for imgs, true_masks, path_img in test_loader:
    model.eval()
    imgs = imgs.to(device=device, dtype=torch.float32)
    mask_type = torch.float32
    true_masks = true_masks.to(device=device, dtype=mask_type)
    print(flag)
    if path_img[0][-10: -7] != flag:
        # if i > 10:
        #     break
        # i += 1
        flag = path_img[0][-10: -7]
        tot.append(sum(tot_sub)/len(tot_sub))
        tot_sub = []
        tot_hsd.append(sum(tot_sub_hsd)/len(tot_sub_hsd))
        tot_sub_hsd = []
    with torch.no_grad():
        reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = model(imgs, true_masks, 'test')

    mask_pred = a_out[:, :4, :, :]
    pred = F.softmax(mask_pred, dim=1)
    pred = (pred > 0.5).float()
    dice = dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
    hsd = hausdorff_distance(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :])
    tot_sub.append(dice)
    tot_sub_hsd.append(hsd)
    print(step)
    step += 1

print(tot)

print(sum(tot)/len(tot))
print(statistics.stdev(tot))

print(tot_hsd)

print(sum(tot_hsd)/len(tot_hsd))
print(statistics.stdev(tot_hsd))
