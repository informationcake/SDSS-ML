# SDSS-ML
The paper: https://arxiv.org/pdf/1909.10963.pdf

Identifying galaxies, quasars and stars with machine learning: a new catalogue of classifications for 111 million SDSS sources without spectra


These scripts explore galaxy/quasar/star classification from optical and infrared data using machine learning. We use SDSS Data Release 15.

SDSS_ML.py
- Cleans data, builds random forest model

SDSS_ML_analysis.py
- Creates analysis plots using the output from SDSS_ML.py

SDSS_ML_plotmaglim.py
- Generates plots from SDSS_ML.py run with magnitude limits on the training dataset

SDSS_ML_knnplots.py
- Finds nearest neighbours in 1-D and 10-D feature spaces and makes plots

SDSS_ML_classifynew.py
- Classifies new sources without spectra, and makes plots assessing the output. 

Our catalogue can be found and referenced under our DOI here: https://www.doi.org/10.5281/zenodo.3459294
