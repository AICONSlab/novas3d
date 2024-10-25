# NOVAS3D

## What NOVAS3D does

*NOVAS3D* (Network of Vessel Analysis in 3D) is a CNN-based pipeline to extarct vascular networks from 3D flourescent microscopy images and track their morphological changes across time.

## How to install NOVAS3D

To install the current release run:

```bash
pip install novas3d
pip install git+https://github.com/Image-Py/sknw.git@18f18ab94794964a6dd7a76dd8a2c5c00dab6fd1
```

## Get started using NOVAS3D

For a tutorial on using the pipeline please see tutorial.ipynb

### Downlaod example data

Ensure [git-lfs](https://git-lfs.com/) is installed and enabled or else pickle files present in the example data and model files will not download properly.

```bash
git clone https://huggingface.co/datasets/mrozak/novas3d_example_data
git clone https://huggingface.co/mrozak/NOVAS3D_Vessel_and_Neuron_Segmentation
```

## How to cite NOVAS3D

If you use NOVAS3D in your work please cite our paper:

Matthew Rozak James Mester Ahmadreza Attarpour Adrienne Dorr Shruti Patel Margaret Koletar Mary Hill JoAnne McLaurin Maged Goubran Bojana Stefanovic 2024 "A Deep Learning Pipeline for Mapping in situ Network-level Neurovascular Coupling in Multi-photon Fluorescence Microscopy" eLife13:RP95525 https://doi.org/10.7554/eLife.95525.2
____________________________

For more details, see our [docs](https://novas3d.readthedocs.io/en/latest/index.html).

