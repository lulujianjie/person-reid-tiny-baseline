.. demo documentation master file, created by
   sphinx-quickstart on Thu Feb  6 12:10:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tiny Person ReID Baseline
=========================

Tiny Person ReID Baseline is an open source code for person re-identification based on PyTorch.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Hyperparameter Configuration

   config/default

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Datasets

   datasets/preprocessing
   datasets/market1501
   

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Loss Function

   loss/triplet

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model

   model/backbones

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Processor

   processor/processor

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Solver

   solver/lr_scheduler
   solver/optimizer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tools

   tools/search_reranking_params
   tools/visualize_data_aug
   tools/visualize_reid

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Utils

