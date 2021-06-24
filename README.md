# Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation
![model](figure/model.png)

This repository contains the official Pytorch implementation of Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation (accepted by [MICCAI 2021](https://miccai2021.org/en/)).

The repository is created by [Xiao Liu](https://github.com/xxxliu95), [Spyridon Thermos](https://github.com/spthermo), [Alison O'Neil](https://www.eng.ed.ac.uk/about/people/dr-alison-oneil), and [Sotirios A. Tsaftaris](https://www.eng.ed.ac.uk/about/people/dr-sotirios-tsaftaris), as a result of the collaboration between [The University of Edinburgh](https://www.eng.ed.ac.uk/) and [Canon Medical Systems Europe](https://eu.medical.canon/).

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

# Datasets
We used two datasets in the paper. [Multi-Centre, Multi-Vendor & Multi-Disease
Cardiac Image Segmentation Challenge (M&Ms) datast](https://www.ub.edu/mnms/) and [Spinal cord grey matter segmentation challenge dataset](http://niftyweb.cs.ucl.ac.uk/challenge/index.php).

# Qualitative results
![results](figure/result.png)

# Citation
TBA

# Acknowlegement
Part of the code is based on [SDNet](https://github.com/spthermo/SDNet), [MLDG](https://github.com/HAHA-DL/MLDG), [medical-mldg-seg](https://github.com/Pulkit-Khandelwal/medical-mldg-seg) and [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

# License
All scripts are released under the MIT License.
