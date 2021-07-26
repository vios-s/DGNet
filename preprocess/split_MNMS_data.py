
import os
import shutil
import xlrd

def walk_path(dir):
    dir = dir
    paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for name in dirs:
            paths.append(os.path.join(root, name))
            # print(paths)
    return paths


######################################################################################################
#Load excel information:
# cell_value(1,0) -> cell_value(175,0)
ex_file = '/home/s1575424/xiao/Year2/mnms_split_data/mnms_dataset_info.xlsx'
wb = xlrd.open_workbook(ex_file)
sheet = wb.sheet_by_index(0)
# sheet.cell_value(r, c)

vendor_A = []
vendor_B = []
vendor_C = []
vendor_D = []

center_2 = []
center_3 = []

for i in range(1, 346):
    if sheet.cell_value(i, 2)=='A':
        vendor_A.append(sheet.cell_value(i, 0))
    elif sheet.cell_value(i, 2)=='B':
        vendor_B.append(sheet.cell_value(i, 0))
        if sheet.cell_value(i, 3)==2:
            center_2.append(sheet.cell_value(i, 0))
        else:
            center_3.append(sheet.cell_value(i, 0))
    elif sheet.cell_value(i, 2) == 'C':
        vendor_C.append(sheet.cell_value(i, 0))
    elif sheet.cell_value(i, 2) == 'D':
        vendor_D.append(sheet.cell_value(i, 0))
    else:
        break

######################################################################################################
# move data to the corresponding folders: mnms_split_data/Labeled/(vendorA, vendorB, vendorC, vendorD), mnms_split_data/Unlabeled/vendorC
path_vendorA = '/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorA/'
path_vendorB = '/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorB/'
path_vendorC = '/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorC/'
path_vendorD = '/home/s1575424/xiao/Year2/mnms_split_data/Labeled/vendorD/'

labeled_train_paths = walk_path('/home/s1575424/xiao/Year2/OpenDataset/Training/Labeled')
testing_paths = walk_path('/home/s1575424/xiao/Year2/OpenDataset/Testing')
val_paths = walk_path('/home/s1575424/xiao/Year2/OpenDataset/Validation')

i = 0
for train_path in labeled_train_paths:
    if train_path[-6:] in vendor_A:
        shutil.move(train_path, path_vendorA)
    elif train_path[-6:] in vendor_B:
        if train_path[-6:] in center_2:
            shutil.move(train_path, path_vendorB + '/center2')
        else:
            shutil.move(train_path, path_vendorB + '/center3')
    else:
        continue
    i += 1
    print(i)

i = 0
for test_path in testing_paths:
    if test_path[-6:] in vendor_A:
        shutil.move(test_path, path_vendorA)
    elif test_path[-6:] in vendor_B:
        if test_path[-6:] in center_2:
            shutil.move(test_path, path_vendorB + '/center2')
        else:
            shutil.move(test_path, path_vendorB + '/center3')
    elif test_path[-6:] in vendor_C:
        shutil.move(test_path, path_vendorC)
    elif test_path[-6:] in vendor_D:
        shutil.move(test_path, path_vendorD)
    else:
        continue
    i += 1
    print(i)

i = 0
for val_path in val_paths:
    if val_path[-6:] in vendor_A:
        shutil.move(val_path, path_vendorA)
    elif val_path[-6:] in vendor_B:
        if val_path[-6:] in center_2:
            shutil.move(val_path, path_vendorB + '/center2')
        else:
            shutil.move(val_path, path_vendorB + '/center3')
    elif val_path[-6:] in vendor_C:
        shutil.move(val_path, path_vendorC)
    elif val_path[-6:] in vendor_D:
        shutil.move(val_path, path_vendorD)
    else:
        continue
    i += 1
    print(i)