# Analysis of macrophage enhancers in response to IL-4
This repository includes codes associated with analysis on the sequencing data for macrophages in response to IL-4. 

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

The well-trained models used in our analysis are available [here](https://drive.google.com/drive/folders/1R0DTxLh9KnejTSo7ugaz2TzxAucVEdeV?usp=sharing).

### Scoring nucleotide importance
The codes for interpreting trained models and assigning single-nucleotide importance scores are stored in [model_interpretation/](https://github.com/zeyang-shen/macrophage_IL4Response/tree/main/model_interpretation)