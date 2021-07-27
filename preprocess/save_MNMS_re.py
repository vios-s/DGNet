import nibabel as nib
import argparse
import os
import numpy as np
from PIL import Image
import configparser
import xlrd

def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def save_mask(img, path, name):
    h = img.shape[0]
    w = img.shape[1]
    img_1 = (img == 1) * 255
    img_2 = (img == 2) * 255
    img_3 = (img == 3) * 255
    img = np.concatenate((img_1.reshape(h,w,1), img_2.reshape(h,w,1), img_3.reshape(h,w,1)), axis=2)
    img  = Image.fromarray(img.astype(np.uint8), 'RGB')
    img.save(os.path.join(path, name))

def save_image(img, path, name):
    if img.min() < 0 or img.max() <= 0:
        pass
    else:
        img -= img.min()
        img  /= img.max()

    img  *= 255
    img  = Image.fromarray(img.astype(np.uint8), 'L')
    img.save(os.path.join(path, name))

def save_np(img, path, name):
    np.savez_compressed(os.path.join(path, name), img)

def save_mask_np(img, path, name):
    h = img.shape[0]
    w = img.shape[1]
    img_1 = (img == 1)
    img_2 = (img == 2)
    img_3 = (img == 3)
    img = np.concatenate((img_1.reshape(h,w,1), img_2.reshape(h,w,1), img_3.reshape(h,w,1)), axis=2)
    np.savez_compressed(os.path.join(path, name), img)


parser = argparse.ArgumentParser()
parser.add_argument('--LabeledVendorA', type=str, default='/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorA/', help='The root of dataset.')

parser.add_argument('--LabeledVendorBcenter2', type=str, default='/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorB/center2/', help='The root of dataset.')
parser.add_argument('--LabeledVendorBcenter3', type=str, default='/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorB/center3/', help='The root of dataset.')

parser.add_argument('--LabeledVendorC', type=str, default='/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorC/', help='The root of dataset.')

parser.add_argument('--LabeledVendorD', type=str, default='/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorD/', help='The root of dataset.')

parser.add_argument('--UnlabeledVendorC', type=str, default='/home/s1575424/xiao/Year2/mnms_split_data/Unlabeled/vendorC/', help='The root of dataset.')

arg = parser.parse_args()

######################################################################################################
#Load excel information:
# cell_value(1,0) -> cell_value(175,0)
ex_file = '/home/s1575424/xiao/Year2/mnms_split_data/mnms_dataset_info.xlsx'
wb = xlrd.open_workbook(ex_file)
sheet = wb.sheet_by_index(0)
# sheet.cell_value(r, c)

# Save data dirs
LabeledVendorA_data_dir = '/home/s1575424/xiao/Year2/mnms_split_2D_re/Labeled/vendorA/'

LabeledVendorB2_data_dir = '/home/s1575424/xiao/Year2/mnms_split_2D_re/Labeled/vendorB/center2/'

LabeledVendorB3_data_dir = '/home/s1575424/xiao/Year2/mnms_split_2D_re/Labeled/vendorB/center3/'

LabeledVendorC_data_dir = '/home/s1575424/xiao/Year2/mnms_split_2D_re/Labeled/vendorC/'

LabeledVendorD_data_dir = '/home/s1575424/xiao/Year2/mnms_split_2D_re/Labeled/vendorD/'

UnlabeledVendorC_data_dir = '/home/s1575424/xiao/Year2/mnms_split_2D_re/Unlabeled/vendorC/'


# Load all the data names
LabeledVendorA_names = sorted(os.listdir(arg.LabeledVendorA))

LabeledVendorB2_names = sorted(os.listdir(arg.LabeledVendorBcenter2))
LabeledVendorB3_names = sorted(os.listdir(arg.LabeledVendorBcenter3))

LabeledVendorC_names = sorted(os.listdir(arg.LabeledVendorC))

LabeledVendorD_names = sorted(os.listdir(arg.LabeledVendorD))

UnlabeledVendorC_names = sorted(os.listdir(arg.UnlabeledVendorC))

#### Output: non-normed, non-cropped, no HW transposed npz file.

######################################################################################################
# Load LabeledVendorA data and save them to 2D images

for num_pat in range(0, len(LabeledVendorA_names)):
    gz_name = sorted(os.listdir(arg.LabeledVendorA+LabeledVendorA_names[num_pat]+'/'))
    patient_root = arg.LabeledVendorA+LabeledVendorA_names[num_pat] + '/' + gz_name[0]
    img = nib.load(patient_root)
    img_np = img.get_fdata()

    header = img.header
    resolution = header.get_zooms()
    X_scan_re = []
    X_scan_re.append(resolution[0])
    X_scan_re_np = np.array(X_scan_re)

    print('patient%03d...' % num_pat)
    save_labeled_data_root = LabeledVendorA_data_dir

    img_np = np.transpose(img_np, (0, 1, 3, 2))

    ## save each image of the patient
    for num_slice in range(img_np.shape[2]*img_np.shape[3]):
        save_np(X_scan_re_np, save_labeled_data_root, '%03d%03d' % (num_pat, num_slice))


######################################################################################################
# Load LabeledVendorB2 data and save them to 2D images
for num_pat in range(0, len(LabeledVendorB2_names)):
    gz_name = sorted(os.listdir(arg.LabeledVendorBcenter2+LabeledVendorB2_names[num_pat]+'/'))
    patient_root = arg.LabeledVendorBcenter2+LabeledVendorB2_names[num_pat] + '/' + gz_name[0]
    img = nib.load(patient_root)
    img_np = img.get_fdata()

    header = img.header
    resolution = header.get_zooms()
    X_scan_re = []
    X_scan_re.append(resolution[0])
    X_scan_re_np = np.array(X_scan_re)


    print('patient%03d...' % num_pat)
    save_labeled_data_root = LabeledVendorB2_data_dir

    img_np = np.transpose(img_np, (0, 1, 3, 2))

    if num_pat == 0:
        print(type(img_np[0,0,0,0]))

    ## save each image of the patient
    for num_slice in range(img_np.shape[2]*img_np.shape[3]):
        save_np(X_scan_re_np, save_labeled_data_root, '%03d%03d' % (num_pat, num_slice))



######################################################################################################
# Load LabeledVendorB3 data and save them to 2D images
for num_pat in range(0, len(LabeledVendorB3_names)):
    gz_name = sorted(os.listdir(arg.LabeledVendorBcenter3+LabeledVendorB3_names[num_pat]+'/'))
    patient_root = arg.LabeledVendorBcenter3+LabeledVendorB3_names[num_pat] + '/' + gz_name[0]
    img = nib.load(patient_root)
    img_np = img.get_fdata()

    header = img.header
    resolution = header.get_zooms()
    X_scan_re = []
    X_scan_re.append(resolution[0])
    X_scan_re_np = np.array(X_scan_re)


    print('patient%03d...' % num_pat)
    save_labeled_data_root = LabeledVendorB3_data_dir

    img_np = np.transpose(img_np, (0, 1, 3, 2))

    if num_pat == 0:
        print(type(img_np[0,0,0,0]))

    ## save each image of the patient
    for num_slice in range(img_np.shape[2]*img_np.shape[3]):
        save_np(X_scan_re_np, save_labeled_data_root, '%03d%03d' % (num_pat, num_slice))


######################################################################################################
# Load LabeledVendorC data and save them to 2D images
for num_pat in range(0, len(LabeledVendorC_names)):
    gz_name = sorted(os.listdir(arg.LabeledVendorC+LabeledVendorC_names[num_pat]+'/'))
    patient_root = arg.LabeledVendorC+LabeledVendorC_names[num_pat] + '/' + gz_name[0]
    img = nib.load(patient_root)
    img_np = img.get_fdata()

    header = img.header
    resolution = header.get_zooms()
    X_scan_re = []
    X_scan_re.append(resolution[0])
    X_scan_re_np = np.array(X_scan_re)

    print('patient%03d...' % num_pat)
    save_labeled_data_root = LabeledVendorC_data_dir

    img_np = np.transpose(img_np, (0, 1, 3, 2))

    if num_pat == 0:
        print(type(img_np[0,0,0,0]))

    ## save each image of the patient
    for num_slice in range(img_np.shape[2]*img_np.shape[3]):
        save_np(X_scan_re_np, save_labeled_data_root, '%03d%03d' % (num_pat, num_slice))


######################################################################################################
# Load LabeledVendorD data and save them to 2D images
for num_pat in range(0, len(LabeledVendorD_names)):
    gz_name = sorted(os.listdir(arg.LabeledVendorD+LabeledVendorD_names[num_pat]+'/'))
    patient_root = arg.LabeledVendorD+LabeledVendorD_names[num_pat] + '/' + gz_name[0]
    img = nib.load(patient_root)
    img_np = img.get_fdata()

    header = img.header
    resolution = header.get_zooms()
    X_scan_re = []
    X_scan_re.append(resolution[0])
    X_scan_re_np = np.array(X_scan_re)

    print('patient%03d...' % num_pat)
    save_labeled_data_root = LabeledVendorD_data_dir

    img_np = np.transpose(img_np, (0, 1, 3, 2))

    if num_pat == 0:
        print(type(img_np[0,0,0,0]))

    ## save each image of the patient
    for num_slice in range(img_np.shape[2]*img_np.shape[3]):
        save_np(X_scan_re_np, save_labeled_data_root, '%03d%03d' % (num_pat, num_slice))


######################################################################################################
# Load UnlabeledVendorC data and save them to 2D images
for num_pat in range(0, len(UnlabeledVendorC_names)):
    gz_name = sorted(os.listdir(arg.UnlabeledVendorC+UnlabeledVendorC_names[num_pat]+'/'))
    patient_root = arg.UnlabeledVendorC+UnlabeledVendorC_names[num_pat] + '/' + gz_name[0]
    img = nib.load(patient_root)
    img_np = img.get_fdata()

    header = img.header
    resolution = header.get_zooms()
    X_scan_re = []
    X_scan_re.append(resolution[0])
    X_scan_re_np = np.array(X_scan_re)

    print('patient%03d...' % num_pat)
    save_labeled_data_root = UnlabeledVendorC_data_dir

    img_np = np.transpose(img_np, (0, 1, 3, 2))

    if num_pat == 0:
        print(type(img_np[0,0,0,0]))

    ## save each image of the patient
    for num_slice in range(img_np.shape[2]*img_np.shape[3]):
        save_np(X_scan_re_np, save_labeled_data_root, '%03d%03d' % (num_pat, num_slice))