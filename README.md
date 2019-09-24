# SDSS-ML

Identifying galaxies, quasars and stars with machine learning: a new catalogue of classifications for 111 million SDSS sources without spectra


These scripts explore galaxy/quasar/star classification from optical and infrared data using machine learning.

SDSS_ML.py
- This script is aimed at cleaning data, and also training, validating and testing models using sources with spectra.

SDSS_ML_analysis.py
- This script is aimed at creating analysis plots using the data from spectroscopic sources.

SDSS_ML_plotmaglim.py
- Generates plots from SDSS_ML.py run with magnitude limits on the training dataset

SDSS_ML_knnplots.py
- Finds nearest neighbours in 1-D and 10-D feature spaces and makes plots

SDSS_ML_classifynew.py
- This script classifies new sources without spectra, and makes plots assessing the output. 

