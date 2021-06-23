# Deep learning analysis of macrophages in response to IL-4
This repository includes codes and data for the analysis of the genetic variation effects on macrophages in response to IL-4 using **deep learning**. The training datasets were based on the sequencing data at [GEO:GSE159630](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159630). If you use our findings, codes, or data, please cite:

[Hoeksema, et al. Mechanisms underlying divergent responses of genetically distinct macrophages to IL-4. Science Advances, 2021.](https://advances.sciencemag.org/content/7/25/eabf9808)

## Deep neural networks
Software dependencies:
* Python 3
* [Keras 2.3.1](https://pypi.org/project/Keras/2.3.1/)
* [tensorflow 2.1.0](https://www.tensorflow.org/install/pip)
* [scikit-learn 0.21.3](https://scikit-learn.org/stable/install.html)
* [deeplift 0.6.10.0](https://github.com/kundajelab/deeplift)
* [biopython 1.76](https://biopython.org/wiki/Download)

### Training neural networks
The codes and data for training and evaluating neural networks are stored in folder: [model_training](https://github.com/zeyang-shen/macrophage_IL4Response/tree/main/model_training)

The configuration files of well-trained models used in the paper are available on [Google Drive](https://drive.google.com/drive/folders/1R0DTxLh9KnejTSo7ugaz2TzxAucVEdeV?usp=sharing).

### Scoring nucleotide importance
The codes for interpreting trained models and assigning single-nucleotide importance scores are stored in folder: [model_interpretation](https://github.com/zeyang-shen/macrophage_IL4Response/tree/main/model_interpretation)
