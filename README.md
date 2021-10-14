# Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation
![model](figures/model.png)

This repository contains the official Pytorch implementation of [Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation](https://arxiv.org/abs/2106.13292)(accepted by [MICCAI 2021](https://miccai2021.org/en/) as Oral).

The repository is created by [Xiao Liu](https://github.com/xxxliu95), [Spyridon Thermos](https://github.com/spthermo), [Alison O'Neil](https://vios.science/team/oneil), and [Sotirios A. Tsaftaris](https://www.eng.ed.ac.uk/about/people/dr-sotirios-tsaftaris), as a result of the collaboration between [The University of Edinburgh](https://www.eng.ed.ac.uk/) and [Canon Medical Systems Europe](https://eu.medical.canon/). You are welcome to visit our group website: [vios.s](https://vios.science/)

# System Requirements
* Pytorch 1.5.1 or higher with GPU support
* Python 3.7.2 or higher
* SciPy 1.5.2 or higher
* CUDA toolkit 10 or newer
* Nibabel
* Pillow
* Scikit-image
* TensorBoard
* Tqdm

# Abstract
Generalising deep models to new data from new centres (termed here domains) remains a challenge. This is largely attributed to shifts in data statistics (domain shifts) between source and unseen domains. Recently, gradient-based meta-learning approaches where the training data are split into meta-train and meta-test sets to simulate and handle the domain shifts during training have shown improved generalisation performance. However, the current fully supervised meta-learning approaches are not scalable for medical image segmentation, where large effort is required to create pixel-wise annotations. Meanwhile, in a low data regime, the simulated domain shifts may not approximate the true domain shifts well across source and unseen domains. To address this problem, we propose a novel semi-supervised meta-learning framework with disentanglement. We explicitly model the representations related to domain shifts. Disentangling the representations and combining them to reconstruct the input image allows unlabeled data to be used to better approximate the true domain shifts for meta-learning. Hence, the model can achieve better generalisation performance, especially when there is a limited amount of labeled data. Experiments show that the proposed method is robust on different segmentation tasks and achieves state-of-the-art generalisation performance on two public benchmarks.

# Training
Note that the hyperparameters in the current version are tuned for BCD to A cases. For other cases, the hyperparameters and few specific layers of the model are slightly different. To train the model with 5% labeled data, run:
```
python train_meta.py -e 150 -c cp_dgnet_meta_5_tvA/ -t A -w DGNetRE_COM_META_5_tvA -g 0
```
Here the defualt learning rate is 4e-5. You can change the learning rate by adding ```-lr xxx```.

To train the model with 100% labeled data, try to change the training parameters to:
``` 
k_un = 1
k1 = 20
k2 = 2
```
The first parameter controls how many interations you want the model to be trained with unlabaled data for every interation of training. ```k1 = 20``` means the learning rate will start to decay after 20 epochs and ```k2 = 2``` means it will check if decay learning every 2 epochs. 

Also, change the ratio ```k=0.05``` (line 221) to ```k=1``` in ```mms_dataloader_meta_split.py```.

Then, run:
```
python train_meta.py -e 80 -c cp_dgnet_meta_100_tvA/ -t A -w DGNetRE_COM_META_100_tvA -g 0
```
Finally, when train the model, changing the ```resampling_rate=1.2``` (line 47) in ```mms_dataloader_meta_split.py``` to 1.1 - 1.3 may cause better results. This will change the rescale ratio when preprocess the images.

# Inference
After training, you can test the model:
```
python inference.py -bs 1 -c cp_dgnet_meta_100_tvA/ -t A -g 0
```
Similarly, changing the ```resampling_rate=1.2``` (line 47) in ```mms_dataloader_meta_split_test.py``` to 1.1 - 1.3 may cause better results.

# Datasets
We used two datasets in the paper: [Multi-Centre, Multi-Vendor & Multi-Disease
Cardiac Image Segmentation Challenge (M&Ms) datast](https://www.ub.edu/mnms/) and [Spinal cord grey matter segmentation challenge dataset](http://niftyweb.cs.ucl.ac.uk/challenge/index.php). The dataloader in this repo is only for M&Ms dataset.

# Qualitative results
![results](figures/result.png)

# Citation
```
@inproceedings{liu2021semi,
  title={Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation},
  author={Liu, Xiao and Thermos, Spyridon and O’Neil, Alison and Tsaftaris, Sotirios A},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={307--317},
  year={2021},
  organization={Springer}
}
```

# Acknowlegement
Part of the code is based on [SDNet](https://github.com/spthermo/SDNet), [MLDG](https://github.com/HAHA-DL/MLDG), [medical-mldg-seg](https://github.com/Pulkit-Khandelwal/medical-mldg-seg) and [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

# License
All scripts are released under the MIT License.
