.. NOVAS3D documentation master file, created by
   sphinx-quickstart on Fri Dec 14 15:34:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NOVAS3D's documentation!
==================================

.. image:: images/ezgif-2-66f21f6470.png
    :width: 550px
    :alt: Graph abstract
    :align: center

*NOVAS3D* is a CNN-based segmentation based pipeline for anayzing vascular network morphology 
and its chnages over time using 3D two-photon flourescent microscopy images from the 
Function Brain Imaging Lab.
It can register images of an abitrary size to a reference frame.
The pipeline is designed to be modular and flexible, allowing for easy integration of new
segmentation models.It outputs a networkx graph representation of the segmented vasculature 
network common to all timepoints. Each vessel segment is represented as an edge in the graph, 
with the nodes as their branching points. The radius of each vessel segment is also stored 
as an edge attribute.

.. image:: images/Computational_pipeline.png
    :width: 550px
    :alt: Graph abstract
    :align: center

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   before_install
   install
   issues


