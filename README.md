# SDSS-ML
Identifying galaxies, quasars and stars with machine learning: a new catalogue of classifications for 111 million SDSS sources without spectra

The paper, published in A&A: https://arxiv.org/abs/1909.10963

Our catalogue can be found and referenced under our DOI here: https://www.doi.org/10.5281/zenodo.3459293

These scripts explore galaxy/quasar/star classification from optical and infrared data using machine learning. We use SDSS Data Release 15. Interestingly, SDSS DR16 has spectroscopic observations of new quasars, which we had already identified using photometric data :)

Here is a 4 slide poster summarising our research, made for the RAS Early Career Poster Exhibition: https://docs.google.com/presentation/d/1r2D9P0JuQGMk2_RhQUQAsk3vunshTjhI22T_OnPPdt4/edit?usp=sharing

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

SDSS_ML_UMAP.py
- Runs UMAP on the spectroscopic and photometric datasets, and makes 2D plots.

When using our work, please consider referencing our paper (in the usual way via the journal) and our catalogue via the DOI https://www.doi.org/10.5281/zenodo.3459293
We have also created a DOI for our code, if you make use of it and wish to reference: https://doi.org/10.5281/zenodo.3855160

Thanks for taking an interest in our work :)
