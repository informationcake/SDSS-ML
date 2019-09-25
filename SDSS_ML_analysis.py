# Written by Alex Clarke - https://github.com/informationcake/SDSS-ML
# Generates plots verifying the results of SDSS_ML.py on the training and test sets.

# Pre-requisits: Run SDSS_ML.py. Obtain the following .pkl files:
# df.pkl
# data_prep_dict_all.pkl
# classes_pred_all.pkl
# classes_pred_all_proba.pkl
# Optional files:
# classes_pred_boss.pkl
# classes_pred_sdss.pkl

# Version numbers
# python: 3.6.1
# pandas: 0.25.0
# numpy: 1.17.0
# scipy: 1.3.1
# matplotlib: 3.1.1
# sklearn: 0.20.0

import os, sys, glob
import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
mpl.use('TKAgg',warn=False, force=True) #set MPL backend.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import pickle
import multiprocessing
import itertools
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import datashader as ds
from datashader.utils import export_image
from datashader.colors import *

# list of functions:
# save_obj
# load_obj
# histvals_o
# histvals
# prepare_classifications
# plot_trainvsf1
# plot_feature_ranking
# plot_basic_hists
# plot_z_hist
# plot_feature_hist
# plot_compare_sets
# plot_error_or_resolved_hist
# plot_histogram_matrix
# plot_histogram_matrix_f1
# plot_metric_curves
# plot_probs_hist
# make_cmap
# plot_probs_hexscatter



# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#Loading/saving data and models directly in their Python format
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Customised histogram function to help plotting (courtesy of Justin Bray)
def histvals_o(a, cumulative=False, **kwargs):
  """Return x/y values for plotting a histogram."""
  counts, bins = np.histogram(a, **kwargs)
  if cumulative==True:
      counts = np.cumsum(counts)
  x = np.concatenate( list(zip( bins[:-1], bins[1:] )) )
  y = np.concatenate( list(zip( counts,    counts   )) )
  x = np.concatenate(( [x[0]], x, [x[-1]] ))
  y = np.concatenate(( [0],    y,  [0]    ))
  # Fudge to make plotting work properly on log scale.
  y = (y > 0)*y + (y <= 0)*1e-0
  return x,y


# Customised histogram function to help plotting on log scale and other aspects (courtesy of Justin Bray)
def histvals(a, logmin=0.0, cumulative=False, **kwargs):
  """Return x/y values for plotting a histogram."""

  if cumulative:
    lims = kwargs.pop('bins', [a.min(), a.max()])
    density = kwargs.pop('density', False)
    assert not kwargs, 'Unprocessed kwargs in histvals.'

    # Reduce length of array, if possible, by combining duplicate values.
    bins,counts = np.unique(a, return_counts=True)
    #counts, bins = np.histogram(a, **kwargs)

    bins = np.concatenate(( [lims[0]], bins, [lims[-1]] ))
    counts = np.concatenate(( [0], np.cumsum(counts) ))

    if density:
      counts = counts*1./counts.max()
  else:
    counts, bins = np.histogram(a, **kwargs)

  x = np.concatenate( list(zip( bins[:-1], bins[1:] )) )
  y = np.concatenate( list(zip( counts,    counts   )) )

  if not cumulative:
    x = np.concatenate(( [x[0]], x, [x[-1]] ))
    y = np.concatenate(( [0],    y,  [0]    ))

  # Fudge to make plotting work properly on log scale.
  if logmin:
    y = (y > 0)*y + (y <= 0)*logmin
  return x,y






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# Prepare data for plotting. Get correct/missed classifications per class as separate dfs
def prepare_classifications(data, classes_pred, ttsplit=True):
    print('Preparing data...')
    # Each dict entry in data is a df, and df indices are integers. Do not use iloc, always use loc to consistently match back to original df.

    if ttsplit==True:
        select_feat = 'features_test'
        select_class = 'classes_test'
    if ttsplit==False:
        select_feat = 'all_features'
        select_class = 'all_classes'

    # Append predicted classes to the data - this is a np array
    data['classes_pred'] = classes_pred


    # Select features for the misclassified objects
    missed_star = data[select_feat][ (data[select_class] != data['classes_pred']) & (data[select_class] == 'STAR') ]
    missed_quasar = data[select_feat][ (data[select_class] != data['classes_pred']) & (data[select_class] == 'QSO') ]
    missed_galaxy = data[select_feat][ (data[select_class] != data['classes_pred']) & (data[select_class] == 'GALAXY') ]

    # Select features for the correctly classified objects
    correct_star = data[select_feat][ (data[select_class] == data['classes_pred']) & (data[select_class] == 'STAR') ]
    correct_quasar = data[select_feat][ (data[select_class] == data['classes_pred']) & (data[select_class] == 'QSO') ]
    correct_galaxy = data[select_feat][ (data[select_class] == data['classes_pred']) & (data[select_class] == 'GALAXY') ]

    # Missed stars predicted as quasars
    missed_star_as_quasar = data[select_feat][(data[select_class] != data['classes_pred']) & (data[select_class] == 'STAR') & (data['classes_pred'] == 'QSO')]
    # Missed stars predicted as galaxies
    missed_star_as_galaxy = data[select_feat][(data[select_class] != data['classes_pred']) & (data[select_class] == 'STAR') & (data['classes_pred'] == 'GALAXY')]

    # Missed quasars predicted as stars
    missed_quasar_as_star = data[select_feat][(data[select_class] != data['classes_pred']) & (data[select_class] == 'QSO') & (data['classes_pred'] == 'STAR')]
    # Missed quasars predicted as galaxies
    missed_quasar_as_galaxy = data[select_feat][(data[select_class] != data['classes_pred']) & (data[select_class] == 'QSO') & (data['classes_pred'] == 'GALAXY')]

    # Missed galaxies predicted as stars
    missed_galaxy_as_star = data[select_feat][(data[select_class] != data['classes_pred']) & (data[select_class] == 'GALAXY') & (data['classes_pred'] == 'STAR')]
    # Missed galaxies predicted as quasars
    missed_galaxy_as_quasar = data[select_feat][(data[select_class] != data['classes_pred']) & (data[select_class] == 'GALAXY') & (data['classes_pred'] == 'QSO')]

    # tests
    #print(df['class'].value_counts())
    #print('-'*30)
    #print(correct_star.index.to_numpy())
    #a = correct_star.index.to_numpy()

    #print(df.loc[missed_star.index.values]['class'].value_counts())
    #print('-'*30)

    return data, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_trainvsf1(f1_f, f1_f_sampleG, p_f, p_f_sampleG, r_f, r_f_sampleG):
    mpl.rcParams.update({'font.size': 10})
    markersize = 3
    # Data saved to disk from SDSS_ML.py
    f1scores = load_obj(f1_f)
    f1scores_sampleG = load_obj(f1_f_sampleG)
    precisions = load_obj(p_f)
    precisions_sampleG = load_obj(p_f_sampleG)
    recalls = load_obj(r_f)
    recalls_sampleG = load_obj(r_f_sampleG)

    train_range = f1scores[0] # saved as first in list from SDSS_ML.py train_vs_f1score()
    f1 = np.array(f1scores[1:]) # ignore first in list
    f1_sampleG = np.array(f1scores_sampleG[1:]) # ignore first in list
    precision = np.array(precisions)
    precision_sampleG = np.array(precisions_sampleG)
    recall = np.array(recalls)
    recall_sampleG = np.array(recalls_sampleG)


    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7,9), sharex=True)
    plt.sca(ax1)
    plt.plot(train_range, precision[:,0], label='Galaxy', marker='o', markersize=markersize, color=galaxy_c)
    plt.plot(train_range, precision[:,1], label='Quasar', marker='o', markersize=markersize, color=quasar_c)
    plt.plot(train_range, precision[:,2], label='Star', marker='o', markersize=markersize, color=star_c)
    plt.plot(train_range, precision_sampleG[:,0], label='Galaxy (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=galaxy_c)
    plt.plot(train_range, precision_sampleG[:,1], label='Quasar (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=quasar_c)
    plt.plot(train_range, precision_sampleG[:,2], label='Star (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=star_c)
    plt.ylabel('Precision')
    #plt.xscale('log')
    plt.ylim(top=1, bottom=0.85)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=True)

    plt.sca(ax2)
    plt.plot(train_range, recall[:,0], label='Galaxy', marker='o', markersize=markersize, color=galaxy_c)
    plt.plot(train_range, recall[:,1], label='Quasar', marker='o', markersize=markersize, color=quasar_c)
    plt.plot(train_range, recall[:,2], label='Star', marker='o', markersize=markersize, color=star_c)
    plt.plot(train_range, recall_sampleG[:,0], label='Galaxy (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=galaxy_c)
    plt.plot(train_range, recall_sampleG[:,1], label='Quasar (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=quasar_c)
    plt.plot(train_range, recall_sampleG[:,2], label='Star (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=star_c)
    plt.ylabel('Recall')
    plt.ylim(top=1, bottom=0.85)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=True)

    plt.sca(ax3)
    plt.plot(train_range, f1[:,0], label='Galaxy', marker='o', markersize=markersize, color=galaxy_c)
    plt.plot(train_range, f1[:,1], label='Quasar', marker='o', markersize=markersize, color=quasar_c)
    plt.plot(train_range, f1[:,2], label='Star', marker='o', markersize=markersize, color=star_c)
    plt.plot(train_range, f1_sampleG[:,0], label='Galaxy (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=galaxy_c)
    plt.plot(train_range, f1_sampleG[:,1], label='Quasar (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=quasar_c)
    plt.plot(train_range, f1_sampleG[:,2], label='Star (trained on balanced classes)', marker='o', markersize=markersize, ls='--', color=star_c)
    plt.xlabel('Fraction of data trained on')
    plt.ylabel('F1-score')
    plt.minorticks_on()
    plt.ylim(top=1, bottom=0.85)
    plt.xlabel('Fraction of the training dataset trained on')
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
    plt.tick_params(axis='y', which='both', right=True)
    plt.legend(frameon=False)
    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    plt.savefig('Train-vs-PRF1.pdf')






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_feature_ranking(pipeline, feature_names):
    clf=pipeline.steps[0][1] #get classifier used
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names_importanceorder=[]
    for f in range(len(indices)):
        #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
        feature_names_importanceorder.append(str(feature_names[indices[f]]))
    # Plot the feature importances of the forest
    mpl.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10,6))
    plt.bar(range(len(indices)), importances[indices],
           color='slategrey', yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='horizontal')
    plt.ylabel('Feature importance')
    plt.ylim(bottom=-0.02)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    #plt.tick_params(axis='y', which='both', right=True)
    #plt.gca().tick_params(axis='y', which='minor', left=True, right=True)
    plt.tight_layout()
    plt.savefig('feature-ranking.pdf')






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_basic_hists(df):
    # histogram each class (STAR/GALAXY/QSO) per instrument (SDSS/BOSS)
    bins = np.linspace(8,26,100)
    linewidth = 1
    plt.subplots(1, 1, figsize=(6,3.5))
    xvar='psf_r_corr'
    x, s = histvals(df[(df.instrument=='SDSS') & (df['class']=='STAR')][xvar].values, bins=bins)
    x, g = histvals(df[(df.instrument=='SDSS') & (df['class']=='GALAXY')][xvar].values, bins=bins)
    x, q = histvals(df[(df.instrument=='SDSS') & (df['class']=='QSO')][xvar].values, bins=bins)
    x, s2 = histvals(df[(df.instrument=='BOSS') & (df['class']=='STAR')][xvar].values, bins=bins)
    x, g2 = histvals(df[(df.instrument=='BOSS') & (df['class']=='GALAXY')][xvar].values, bins=bins)
    x, q2 = histvals(df[(df.instrument=='BOSS') & (df['class']=='QSO')][xvar].values, bins=bins)
    plt.plot(x, s, label='SDSS Stars: {0:.0f}'.format(np.sum(s)/2), ls='--', dashes=(3,1), color=star_c, linewidth=linewidth)
    plt.plot(x, g, label='SDSS Galaxies: {0:.0f}'.format(np.sum(g)/2), ls='--', dashes=(3,1), color=galaxy_c, linewidth=linewidth)
    plt.plot(x, q, label='SDSS Quasars: {0:.0f}'.format(np.sum(q)/2), ls='--', dashes=(3,1), color=quasar_c, linewidth=linewidth)
    plt.plot(x, s2, label='BOSS Stars: {0:.0f}'.format(np.sum(s2)/2), ls=':', color=star_c, linewidth=1.5)
    plt.plot(x, g2, label='BOSS Galaxies: {0:.0f}'.format(np.sum(g2)/2), ls=':', color=galaxy_c, linewidth=1.5)
    plt.plot(x, q2, label='BOSS Quasars: {0:.0f}'.format(np.sum(q2)/2), ls=':', color=quasar_c, linewidth=1.5)
    plt.plot(x, s+s2, label='All Stars: {0:.0f}'.format(np.sum(s+s2)/2), ls='-', color=star_c, linewidth=1.5)
    plt.plot(x, g+g2, label='All Galaxies: {0:.0f}'.format(np.sum(g+g2)/2), ls='-', color=galaxy_c, linewidth=1.5)
    plt.plot(x, q+q2, label='All Quasars: {0:.0f}'.format(np.sum(q+q2)/2), ls='-', color=quasar_c, linewidth=1.5)
    plt.yscale('log')
    plt.xlabel('PSF r magnitude')
    plt.ylabel('Number of sources')
    plt.minorticks_on()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('All-hist-rmag.pdf')






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Plot histogram of z
def plot_z_hist(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy,
    missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, plot_data_label='unknown'):
    mpl.rcParams.update({'font.size': 10})
    figsize = (7,9) # when there are three subplots
    bins = np.linspace(0, 10 ,100)
    xlab = 'redshift'
    val = 'z'
    df['z'] = df['z'].abs() #forcing stars to have positive redshifts for the purpose of this plot
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ## ------ ------ Galaxy / Quasar ------ ------

    # galaxies misclassified as quasars:
    plt.sca(ax3) # plt.sca gives easier access to parms I find, rather than ax1.
    x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x1, y1, label='missed galaxies as quasars: {0}'.format(len(missed_galaxy_as_quasar)), ls=missed_ls, color=galaxy_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x2, y2, label='missed quasars as galaxies: {0}'.format(len(missed_quasar_as_galaxy)), ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x3, y3, label='correct galaxies: {0}'.format(len(correct_galaxy)), ls=correct_ls, color=galaxy_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x4, y4, label='correct quasars: {0}'.format(len(correct_quasar)), ls=correct_ls, color=quasar_c)
    plt.legend(frameon=False, fontsize=8)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylim(top=3*10**5)
    plt.xlim(-0.19, 7.19)
    plt.minorticks_on()
    plt.ylabel('Number of galaxies/quasars')
    plt.xlabel(xlab)
    plt.tick_params(axis='x', which='both', bottom=True, top=True)
    plt.tick_params(axis='y', which='both', right=True)

    ## ------ ------ Star / Quasar  ------ ------

    plt.sca(ax1)
    x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x1, y1, label='missed stars as quasars: {0}'.format(len(missed_star_as_quasar)), ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x2, y2, label='missed quasars as stars: {0}'.format(len(missed_quasar_as_star)), ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x3, y3, label='correct stars: {0}'.format(len(correct_star)), ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x4, y4, label='correct quasars: {0}'.format(len(correct_quasar)), ls=correct_ls, color=quasar_c)
    plt.legend(frameon=False, fontsize=8)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylim(top=3*10**5)
    plt.xlim(-0.19, 7.19)
    plt.minorticks_on()
    plt.ylabel('Number of quasars/stars')
    plt.xlabel(xlab)
    plt.tick_params(axis='x', which='both', bottom=True, top=True)
    plt.tick_params(axis='y', which='both', right=True)

    ## ------ ------ Star / Galaxy ------ ------

    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x1, y1, label='missed stars as galaxies: {0}'.format(len(missed_star_as_galaxy)), ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_galaxy_as_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x2, y2, label='missed galaxies as stars: {0}'.format(len(missed_galaxy_as_star)), ls=missed_ls, color=galaxy_c)
    x3, y3 = histvals(df.loc[correct_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x3, y3, label='correct stars: {0}'.format(len(correct_star)), ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x4, y4, label='correct galaxies: {0}'.format(len(correct_galaxy)), ls=correct_ls, color=galaxy_c)
    plt.legend(frameon=False, fontsize=8)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylim(top=3*10**5)
    plt.xlim(-0.19, 7.19)
    plt.minorticks_on()
    plt.ylabel('Number of stars/galaxies')
    plt.xlabel(xlab)
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
    plt.tick_params(axis='y', which='both', right=True)
    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    f.savefig('z-hist'+plot_data_label+'.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Compare pairs of classes, plus histogram of resolved status. Re-run with different input for BOSS / SDSS / ALL. Change plot_label each time to keep track of the figures produced.
def plot_feature_hist(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy,
    missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, plot_data_label='unknown'):
    print('Making feature histogram plot... features-.pdf')
    # define labels for the x axis
    xlabels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4'] # in case we want more concise labels.
    # Only keep magnitude features to plot (remove resolvedr feature):
    # maybe this should be an argument in more general terms...
    plot_columns = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4']
    # If you want a plot of cmodel magnitudes use this instead:
    # plot_columns = ['cmod_u_corr', 'cmod_g_corr', 'cmod_r_corr', 'cmod_i_corr', 'cmod_z_corr', 'w1', 'w2', 'w3', 'w4']

    figsize = (10,3.5) # when there are two subplots
    #figsize = (15,4) # when there are three subplots
    linewidth = 1.2
    elinewidth = 0.5
    capthick = 0.5
    correct_capsize = 4
    missed_capsize = 1

    ## ------ ------ Star / Quasar  ------ ------

    # Real stars misclassified as quasars:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plt.sca(ax1) # So if you want separate plots it's easier to comment out required lines. also plt. gives easier access to some parms
    # set up transform to offset each of the lines so they dono't overlap
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData

    a = plt.errorbar(plot_columns, df.loc[missed_star_as_quasar.index.values][plot_columns].mean(), yerr=df.loc[missed_star_as_quasar.index.values][plot_columns].std(), color=star_c, ls=missed_ls, capsize=missed_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='missed stars as quasars: {0}'.format(len(missed_star_as_quasar)), transform=trans+offset(-3))
    a[-1][0].set_linestyle('--')
    b = plt.errorbar(plot_columns, df.loc[missed_quasar_as_star.index.values][plot_columns].mean(), yerr=df.loc[missed_quasar_as_star.index.values][plot_columns].std(), color=quasar_c, ls=missed_ls, capsize=missed_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='missed quasars as stars: {0}'.format(len(missed_quasar_as_star)), transform=trans+offset(+1))
    b[-1][0].set_linestyle('--')
    plt.errorbar(plot_columns, df.loc[correct_star.index.values][plot_columns].mean(), color=star_c, ls=correct_ls, yerr=df.loc[correct_star.index.values][[*plot_columns]].std(), capsize=correct_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='correct stars: {0}'.format(len(correct_star)), transform=trans+offset(-1))
    plt.errorbar(plot_columns, df.loc[correct_quasar.index.values][plot_columns].mean(), color=quasar_c, ls=correct_ls, yerr=df.loc[correct_quasar.index.values][plot_columns].std(), capsize=correct_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='correct quasars: {0}'.format(len(correct_quasar)), transform=trans+offset(+3))


    plt.legend(frameon=False)
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually:
    plt.xlabel('Feature name')
    plt.ylabel('Magnitude')
    plt.ylim(7,25)
    #ax1.margins(x=0.4)
    #ax1.use_sticky_edges = False
    #ax1.autoscale_view(scalex=True)
    plt.tight_layout()


    # Histogram of psf_r - cmod_r. Resolved source or not?
    # Searching the original df because if we don't use resolvedr as a feature it wont be in correct_ and missed_ dataframes.
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x1, y1, label='missed stars as quasars', ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_star.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x2, y2, label='missed quasars as stars', ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_star.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c)
    #plt.legend(frameon=False)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('| PSF r - cmodel r | [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('features-star-quasar'+plot_data_label+'.pdf')

    '''
    # Histogram of errors, high error sources classified worse?
    # Searching the original df because we don't use errors as a feature (they don't improve classification results)
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x1, y1, label='missed stars as quasars', ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_star.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x2, y2, label='missed quasars as stars', ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_star.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c)
    #plt.legend(frameon=False)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(top=2*10**5)
    plt.xlabel('PSF r error [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('star-quasar-'+plot_data_label+'.pdf')
    '''

    ## ------ ------ Star / Galaxy ------ ------

    # Real stars misclassified as galaxies:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plt.sca(ax1) # So if you want separate plots it's easier to comment out required lines. also plt. gives easier access to some parms
    # set up transform to offset each of the lines so they dono't overlap
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData

    a = plt.errorbar(plot_columns, df.loc[missed_star_as_galaxy.index.values][plot_columns].mean(), yerr=df.loc[missed_star_as_galaxy.index.values][plot_columns].std(), color=star_c, ls=missed_ls, capsize=missed_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='missed stars as galaxies: {0}'.format(len(missed_star_as_galaxy)), transform=trans+offset(-3))
    a[-1][0].set_linestyle('--')
    b = plt.errorbar(plot_columns, df.loc[missed_galaxy_as_star.index.values][plot_columns].mean(), yerr=df.loc[missed_galaxy_as_star.index.values][plot_columns].std(), color=galaxy_c, ls=missed_ls, capsize=missed_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='missed galaxies as stars: {0}'.format(len(missed_galaxy_as_star)), transform=trans+offset(+1))
    b[-1][0].set_linestyle('--')
    plt.errorbar(plot_columns, df.loc[correct_star.index.values][plot_columns].mean(), color=star_c, ls=correct_ls, yerr=df.loc[correct_star.index.values][[*plot_columns]].std(), capsize=correct_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='correct stars: {0}'.format(len(correct_star)), transform=trans+offset(-1))
    plt.errorbar(plot_columns, df.loc[correct_galaxy.index.values][plot_columns].mean(), color=galaxy_c, ls=correct_ls, yerr=df.loc[correct_galaxy.index.values][plot_columns].std(), capsize=correct_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='correct galaxies: {0}'.format(len(correct_galaxy)), transform=trans+offset(+3))

    plt.legend(frameon=False, loc='lower left')
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually:
    plt.xlabel('Feature name')
    plt.ylabel('Magnitude')
    plt.ylim(7,25)
    plt.tight_layout()

    # Histogram of psf_r - cmod_r. Resolved source or not?

    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x1, y1, label='missed stars as galaxies', ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_galaxy_as_star.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x2, y2, label='missed galaxies as stars', ls=missed_ls, color=galaxy_c)
    x3, y3 = histvals(df.loc[correct_star.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_galaxy.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x4, y4, label='correct galaxies', ls=correct_ls, color=galaxy_c)
    #plt.legend(frameon=False)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('| PSF r - cmodel r | [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('features-star-galaxy'+plot_data_label+'.pdf')

    '''
    # Histogram of errors, high error sources classified worse?
    # Searching the original df because we don't use errors as a feature (they don't improve classification results)
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x1, y1, label='missed stars as galaxies', ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_galaxy_as_star.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x2, y2, label='missed galaxies as stars', ls=missed_ls, color=galaxy_c)
    x3, y3 = histvals(df.loc[correct_star.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_galaxy.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x4, y4, label='correct galaxies', ls=correct_ls, color=galaxy_c)
    #plt.legend(frameon=False)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(top=2*10**5)
    plt.xlabel('PSF r error [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('star-galaxy_'+plot_data_label+'.pdf')
    '''

    ## ------ ------ Galaxy / Quasar ------ ------

    # Real galaxies misclassified as quasars:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plt.sca(ax1) # So if you want separate plots it's easier to comment out required lines
    # set up transform to offset each of the lines so they dono't overlap
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData

    a = plt.errorbar(plot_columns, df.loc[missed_galaxy_as_quasar.index.values][plot_columns].mean(), yerr=df.loc[missed_galaxy_as_quasar.index.values][plot_columns].std(), color=galaxy_c, ls=missed_ls, capsize=missed_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='missed galaxies as quasars: {0}'.format(len(missed_galaxy_as_quasar)), transform=trans+offset(-3))
    a[-1][0].set_linestyle('--')
    b = plt.errorbar(plot_columns, df.loc[missed_quasar_as_galaxy.index.values][plot_columns].mean(), yerr=df.loc[missed_quasar_as_galaxy.index.values][plot_columns].std(), color=quasar_c, ls=missed_ls, capsize=missed_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='missed quasars as galaxies: {0}'.format(len(missed_quasar_as_galaxy)), transform=trans+offset(+1))
    b[-1][0].set_linestyle('--')
    plt.errorbar(plot_columns, df.loc[correct_galaxy.index.values][plot_columns].mean(), color=galaxy_c, ls=correct_ls, yerr=df.loc[correct_galaxy.index.values][plot_columns].std(), capsize=correct_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='correct galaxies: {0}'.format(len(correct_galaxy)), transform=trans+offset(-1))
    plt.errorbar(plot_columns, df.loc[correct_quasar.index.values][plot_columns].mean(), color=quasar_c, ls=correct_ls, yerr=df.loc[correct_quasar.index.values][plot_columns].std(), capsize=correct_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='correct quasars: {0}'.format(len(correct_quasar)), transform=trans+offset(+3))

    plt.legend(frameon=False)
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually:
    plt.xlabel('Feature name')
    plt.ylabel('Magnitude')
    plt.ylim(7,25)
    plt.tight_layout()


    # Histogram of psf_r - cmod_r. Resolved source or not?
    plt.sca(ax2)
    bins=bins_mag
    x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x1, y1, label='missed galaxies as quasars', ls=missed_ls, color=galaxy_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_galaxy.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x2, y2, label='missed quasars as galaxies', ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_galaxy.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x3, y3, label='correct galaxies', ls=correct_ls, color=galaxy_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values].resolvedr.values, bins=bins_res, density=density)
    plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c)
    #plt.legend(frameon=False)
    plt.yscale('log')
    plt.xscale('log')
    #plt.xlabel('| psf_r - cmodel_r | [magnitude]')
    plt.xlabel('| PSF r - cmodel r | [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('features-galaxy-quasar'+plot_data_label+'.pdf')

    '''
    # Histogram of errors, high error sources classified worse?
    # Searching the original df because we don't use errors as a feature (they don't improve classification results)
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x1, y1, label='missed galaxies as quasars', ls=missed_ls, color=galaxy_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_galaxy.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x2, y2, label='missed quasars as galaxies', ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_galaxy.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x3, y3, label='correct galaxies', ls=correct_ls, color=galaxy_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values].psferr_r.values, bins=bins_err, density=density)
    plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c)
    #plt.legend(frameon=False)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(top=2*10**5)
    plt.xlabel('PSF r error [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('galaxy-quasar_'+plot_data_label+'.pdf')
    '''






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # compare all star/galaxy/quasar for SDSS/BOSS
def plot_compare_sets(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star2, missed_quasar2, missed_galaxy2, correct_star2, correct_quasar2, correct_galaxy2, plot_data_label='unknown'):

    # Wavelength (feature) vs magnitude plots averaging magnitudes for objects
    # Compare over all classes -- Function requires both SDSS and BOSS (or two arbitrary sets)

    # define labels for the x axis
    xlabels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4'] # in case we want more concise labels.
    # Only keep magnitude features to plot (remove resolvedr feature):
    plot_columns = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z', 'w1', 'w2', 'w3', 'w4']
    # If you want a plot of cmodel magnitudes use this instead:
    # plot_columns = ['cmod_u_corr', 'cmod_g_corr', 'cmod_r_corr', 'cmod_i_corr', 'cmod_z', 'w1', 'w2', 'w3', 'w4']

    try:
        correct_star = correct_star[[*plot_columns]]
        missed_star = missed_star[[*plot_columns]]
        correct_galaxy = correct_galaxy[[*plot_columns]]
        missed_galaxy = missed_galaxy[[*plot_columns]]
        correct_quasar = correct_quasar[[*plot_columns]]
        missed_quasar = missed_quasar[[*plot_columns]]

        missed_star_as_quasar = missed_star_as_quasar[[*plot_columns]]
        missed_star_as_galaxy = missed_star_as_galaxy[[*plot_columns]]
        missed_galaxy_as_quasar = missed_galaxy_as_quasar[[*plot_columns]]
        missed_galaxy_as_star = missed_galaxy_as_star[[*plot_columns]]
        missed_quasar_as_star = missed_quasar_as_star[[*plot_columns]]
        missed_quasar_as_galaxy = missed_quasar_as_galaxy[[*plot_columns]]
    except:
        print('Failed - did you remove psf magnitudes from the features? Plots are built to use psf magnitudes. Look in "plots_compare_sets" to adjust the default columns used for these plots')
        exit()

    # ------ Stars ------
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.sca(ax1) # So if you want separate plots it's easier to comment out required lines
    missed_star.mean().plot.line(color=star_c, ls=missed_ls, yerr=missed_star.std(), capsize=missed_capsize, linewidth=linewidth, label='missed star BOSS')
    correct_star.mean().plot.line(color=star_c, ls=correct_ls, yerr=correct_star.std(), capsize=correct_capsize, linewidth=linewidth, label='correct star BOSS')
    missed_star2.mean().plot.line(color=star_c2, ls=missed_ls, yerr=missed_star2.std(), capsize=missed_capsize, linewidth=linewidth, label='missed star SDSS')
    correct_star2.mean().plot.line(color=star_c2, ls=correct_ls, yerr=correct_star2.std(), capsize=correct_capsize, linewidth=linewidth, label='correct star SDSS')
    plt.legend()
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually:

    # - Histogram -
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x1, y1, label='missed stars in BOSS', ls=missed_ls, color=star_c, linewidth=linewidth)
    x2, y2 = histvals(df.loc[correct_star.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x2, y2, label='correct stars in BOSS', ls=correct_ls, color=star_c, linewidth=linewidth)
    x3, y3 = histvals(df.loc[missed_star2.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x3, y3, label='missed stars in SDSS', ls=missed_ls, color=star_c2, linewidth=linewidth)
    x4, y4 = histvals(df.loc[correct_star2.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x4, y4, label='correct stars in SDSS', ls=correct_ls, color=star_c2, linewidth=linewidth)
    #plt.legend(frameon=False)
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('psf_r [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('star-'+plot_data_label+'.pdf')
    if interactive == True:
        plt.show()

    # ------ Quasars ------
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.sca(ax1) # So if you want separate plots it's easier to comment out required lines
    missed_quasar.mean().plot.line(color=quasar_c, ls=missed_ls, yerr=missed_quasar.std(), capsize=missed_capsize, linewidth=linewidth, label='missed quasar BOSS')
    correct_quasar.mean().plot.line(color=quasar_c, ls=correct_ls, yerr=correct_quasar.std(), capsize=correct_capsize, linewidth=linewidth, label='correct quasar BOSS')
    missed_quasar2.mean().plot.line(color=quasar_c2, ls=missed_ls, yerr=missed_quasar2.std(), capsize=missed_capsize, linewidth=linewidth, label='missed quasar SDSS')
    correct_quasar2.mean().plot.line(color=quasar_c2, ls=correct_ls, yerr=correct_quasar2.std(), capsize=correct_capsize, linewidth=linewidth, label='correct quasar SDSS')
    plt.legend()
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually:

    # - Histogram -
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_quasar.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x1, y1, label='missed stars in BOSS', ls=missed_ls, color=quasar_c, linewidth=linewidth)
    x2, y2 = histvals(df.loc[correct_quasar.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x2, y2, label='correct stars in BOSS', ls=correct_ls, color=quasar_c, linewidth=linewidth)
    x3, y3 = histvals(df.loc[missed_quasar2.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x3, y3, label='missed stars in SDSS', ls=missed_ls, color=quasar_c2, linewidth=linewidth)
    x4, y4 = histvals(df.loc[correct_quasar2.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x4, y4, label='correct stars in SDSS', ls=correct_ls, color=quasar_c2, linewidth=linewidth)
    #plt.legend(frameon=False)
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('psf_r [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('quasar-'+plot_data_label+'.pdf')
    if interactive == True:
        plt.show()


    # ------ Galaxies ------
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.sca(ax1) # So if you want separate plots it's easier to comment out required lines
    missed_galaxy.mean().plot.line(color=galaxy_c, ls=missed_ls, yerr=missed_galaxy.std(), capsize=missed_capsize, linewidth=linewidth, label='missed galaxy BOSS')
    correct_galaxy.mean().plot.line(color=galaxy_c, ls=correct_ls, yerr=correct_galaxy.std(), capsize=correct_capsize, linewidth=linewidth, label='correct galaxy BOSS')
    missed_galaxy2.mean().plot.line(color=galaxy_c2, ls=missed_ls, yerr=missed_galaxy2.std(), capsize=missed_capsize, linewidth=linewidth, label='missed galaxy SDSS')
    correct_galaxy2.mean().plot.line(color=galaxy_c2, ls=correct_ls, yerr=correct_galaxy2.std(), capsize=correct_capsize, linewidth=linewidth, label='correct galaxy SDSS')
    plt.legend()
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually:

    # - Histogram -
    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_galaxy.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x1, y1, label='missed galaxies in BOSS', ls=missed_ls, color=galaxy_c, linewidth=linewidth)
    x2, y2 = histvals(df.loc[correct_galaxy.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x2, y2, label='correct galaxies in BOSS', ls=correct_ls, color=galaxy_c, linewidth=linewidth)
    x3, y3 = histvals(df.loc[missed_galaxy2.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x3, y3, label='missed galaxies in SDSS', ls=missed_ls, color=galaxy_c2, linewidth=linewidth)
    x4, y4 = histvals(df.loc[correct_galaxy2.index.values].psf_r.values, bins=bins_mag, density=density)
    plt.plot(x4, y4, label='correct galaxies in SDSS', ls=correct_ls, color=galaxy_c2, linewidth=linewidth)
    #plt.legend(frameon=False)
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('psf_r [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('galaxy'+plot_data_label+'.pdf')
    if interactive == True:
        plt.show()






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Plot histogram of psferr_r or resolvedr (psf_r - cmod_r) status of correct/missed objects per class.
def plot_error_or_resolved_hist(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy,
    missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, val='psferr_r', plot_data_label='unknown'):

    # Searching the original df because if we don't use psferr_r or resolvedr as a feature it wont be in correct_ and missed_ dataframes.

    #figsize = (10,4) # when there are two subplots
    figsize = (15,4) # when there are three subplots
    if val == 'resolvedr':
        bins = bins_res
        xlab = '| PSF r - cmodel r | [magnitude]'
    if val == 'psferr_r':
        bins = bins_err
        xlab =  'PSF r error [magnitude]'


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ## ------ ------ Galaxy / Quasar ------ ------

    # Real galaxies misclassified as quasars:
    plt.sca(ax1) # plt.sca gives easier access to parms I find, rather than ax1.
    x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x1, y1, label='missed galaxies as quasars: {0}'.format(len(missed_galaxy_as_quasar)), ls=missed_ls, color=galaxy_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x2, y2, label='missed quasars as galaxies: {0}'.format(len(missed_quasar_as_galaxy)), ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x3, y3, label='correct galaxies: {0}'.format(len(correct_galaxy)), ls=correct_ls, color=galaxy_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x4, y4, label='correct quasars: {0}'.format(len(correct_quasar)), ls=correct_ls, color=quasar_c)
    plt.legend(frameon=False, fontsize=8)
    plt.yscale('log')
    plt.xscale('log')
    if val == 'psferr_r':
        plt.xlim(left=10**-3)
    plt.ylim(top=8*10**5)
    plt.xlabel(xlab)
    plt.ylabel('Number')
    plt.tight_layout()

    ## ------ ------ Star / Quasar  ------ ------

    plt.sca(ax2)
    x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x1, y1, label='missed stars as quasars: {0}'.format(len(missed_star_as_quasar)), ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_quasar_as_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x2, y2, label='missed quasars as stars: {0}'.format(len(missed_quasar_as_star)), ls=missed_ls, color=quasar_c)
    x3, y3 = histvals(df.loc[correct_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x3, y3, label='correct stars: {0}'.format(len(correct_star)), ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_quasar.index.values][val].values, bins=bins, density=density)
    plt.plot(x4, y4, label='correct quasars: {0}'.format(len(correct_quasar)), ls=correct_ls, color=quasar_c)
    plt.legend(frameon=False, fontsize=8)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(top=8*10**5)
    plt.xlabel(xlab)
    plt.ylabel('Number')
    plt.tight_layout()

    ## ------ ------ Star / Galaxy ------ ------

    plt.sca(ax3)
    x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x1, y1, label='missed stars as galaxies: {0}'.format(len(missed_star_as_galaxy)), ls=missed_ls, color=star_c)
    x2, y2 = histvals(df.loc[missed_galaxy_as_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x2, y2, label='missed galaxies as stars: {0}'.format(len(missed_galaxy_as_star)), ls=missed_ls, color=galaxy_c)
    x3, y3 = histvals(df.loc[correct_star.index.values][val].values, bins=bins, density=density)
    plt.plot(x3, y3, label='correct stars: {0}'.format(len(correct_star)), ls=correct_ls, color=star_c)
    x4, y4 = histvals(df.loc[correct_galaxy.index.values][val].values, bins=bins, density=density)
    plt.plot(x4, y4, label='correct galaxies: {0}'.format(len(correct_galaxy)), ls=correct_ls, color=galaxy_c)
    plt.legend(frameon=False, fontsize=8)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(top=8*10**5)
    plt.xlabel(xlab)
    plt.ylabel('Number')
    plt.tight_layout()

    if val == 'resolvedr':
        f.savefig('resolvedr-hist'+plot_data_label+'.pdf')
    if val == 'psferr_r':
        f.savefig('errorPSFr-hist'+plot_data_label+'.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Histogram matrix plot for all features. Re-run with different input for BOSS / SDSS / ALL
def plot_histogram_matrix(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, plot_data_label='unknown'):

    # histogram over these values:
    mag_vals = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4']
    #mag_vals = ['cmod_u_corr', 'cmod_g_corr', 'cmod_r_corr', 'cmod_i_corr', 'cmod_z_corr', 'w1', 'w2', 'w3', 'w4']
    '''
    if mag_vals[0] in missed_star.columns:
        print('Making histogram over these features: {0}'.format(mag_vals))
    else:
        print('you are not using extinction correct psf magnitudes (perhaps cmodel mags or no extinction correction?). \nYou need to change this by hand at the start of the "plots_histogram_matrix" function.')
        exit()
    '''
    # concise axis labels? (assuming exact magnitude use is put in the caption)
    xlabels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4']
    f, axs = plt.subplots(3, 9, figsize=(12,8), sharey=True, sharex=False)
    ugriz_lim = [12,28]
    w1234_lim = [2,19]
    all_ylim = [0,1e5]
    hist_ls = 'dotted'

    # Note, we are searching the original dataframe for matching indices to get magnitudes for the histogram. Whilst this is not needed (and takes longer), it leave this robust for tests such as removing magnitude bands from features, but still creating a histogram over that magnitude band. I think this is worth the extra 30 seconds wait.

    ## ------ ------ Star / Quasar ------ ------
    # UGRIZ
    for ax, mag_val in zip(axs[0,0:5], mag_vals[0:5]):
        plt.sca(ax)
        x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x1, y1, label='missed stars as quasars', ls=missed_ls, color=star_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x2, y2, label='missed quasars as stars', ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
        x3, y3 = histvals(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
        x4, y4 = histvals(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        plt.yscale('log')
        plt.xlim(ugriz_lim)
        #plt.ylim(all_ylim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()
        if (mag_val=='psf_u_corr') or (mag_val=='cmod_u_corr'):
            plt.ylabel('Number of stars/quasars')
            plt.legend(frameon=False, fontsize=5, loc='upper left')

    # WISE
    for ax, mag_val in zip(axs[0,5:9], mag_vals[5:9]):
        plt.sca(ax)
        x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x1, y1, label='missed stars as quasars', ls=missed_ls, color=star_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x2, y2, label='missed quasars as stars', ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
        x3, y3 = histvals(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
        x4, y4 = histvals(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        plt.yscale('log')
        plt.xlim(w1234_lim)
        #plt.ylim(all_ylim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()

    ## ------ ------ Star / Galaxy ------ ------
    # UGRIZ
    for ax, mag_val in zip(axs[1,0:5], mag_vals[0:5]):
        plt.sca(ax)
        x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x1, y1, label='missed stars as galaxies', ls=missed_ls, color=star_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x2, y2, label='missed galaxies as stars', ls=missed_ls, color=galaxy_c,linewidth=linewidth_hist)
        x3, y3 = histvals(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
        x4, y4 = histvals(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x4, y4, label='correct galaxies', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
        plt.yscale('log')
        plt.xlim(ugriz_lim)
        #plt.ylim(all_ylim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()
        if (mag_val=='psf_u_corr') or (mag_val=='cmod_u_corr'):
            plt.ylabel('Number of stars/galaxies')
            plt.legend(frameon=False, fontsize=5, loc='upper left')

    # WISE
    for ax, mag_val in zip(axs[1,5:9], mag_vals[5:9]):
        plt.sca(ax)
        x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x1, y1, label='missed stars as galaxies', ls=missed_ls, color=star_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x2, y2, label='missed galaxies as stars', ls=missed_ls, color=galaxy_c,linewidth=linewidth_hist)
        x3, y3 = histvals(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
        x4, y4 = histvals(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x4, y4, label='correct galaxies', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
        plt.yscale('log')
        plt.xlim(w1234_lim)
        #plt.ylim(all_ylim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()

    ## ------ ------ Galaxy / Quasar ------ ------
    # UGRIZ
    for ax, mag_val in zip(axs[2,0:5], mag_vals[0:5]):
        plt.sca(ax)
        x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x1, y1, label='missed galaxies as quasars', ls=missed_ls, color=galaxy_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x2, y2, label='missed quasars as galaxies', ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
        x3, y3 = histvals(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x3, y3, label='correct stars', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
        x4, y4 = histvals(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        #plt.legend(frameon=False)
        plt.yscale('log')
        plt.xlim(ugriz_lim)
        #plt.ylim(all_ylim)
        #plt.xlabel(mag_val)
        plt.xlabel(mag_val[4:5]) # replace axis labels. 4:5 removes 'psf_'
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.minorticks_on()
        if (mag_val=='psf_u_corr') or (mag_val=='cmod_u_corr'):
            plt.ylabel('Number of galaxies/quasars')
            plt.legend(frameon=False, fontsize=5, loc='upper left')

    # WISE
    for ax, mag_val in zip(axs[2,5:9], mag_vals[5:9]):
        plt.sca(ax)
        x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x1, y1, label='missed galaxies as quasars', ls=missed_ls, color=galaxy_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x2, y2, label='missed quasars as galaxies', ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
        x3, y3 = histvals(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x3, y3, label='correct galaxies', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
        x4, y4 = histvals(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        plt.plot(x4, y4, label='correct quasars', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        #plt.legend(frameon=False)
        plt.yscale('log')
        plt.xlim(w1234_lim)
        #plt.ylim(all_ylim)
        plt.xlabel(mag_val)
        plt.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.minorticks_on()

    f.suptitle(plot_data_label, fontsize=7, y=0.995)
    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    f.savefig('histmatrix-mag'+plot_data_label+'.pdf', dpi=500)






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_histogram_matrix_f1(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, plot_data_label='unknown', autoscale=True, plot_prob=True):
    # Histogram of precision, recall and F1-score per band. Plus probabilities in bottom row. Plus number histogram to guide eye in top row. This plot, in all its glory, can take a few minutes to make because we are searching the original dataframe for matching indices (lots of df.loc happening) to get magnitudes for the histogram. Whilst many of these df.loc could be avoided, it leaves this function robust for tests such as removing magnitude bands from features, but still creating a histogram over that magnitude band. I think flexibility is worth the extra wait.

    # histogram over these values:
    mag_vals = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4']
    #mag_vals = ['cmod_u_corr', 'cmod_g_corr', 'cmod_r_corr', 'cmod_i_corr', 'cmod_z_corr', 'w1', 'w2', 'w3', 'w4']
    #mag_vals = ['psf_r_corr_u', 'psf_r_corr_g', 'psf_r_corr_i', 'psf_r_corr_z', 'psf_r_corr_w1', 'psf_r_corr_w2', 'psf_r_corr_w3', 'psf_r_corr_w4']

    # Even if you train on different features, this function always histograms over the 9 wavebands. Verbal check:
    print('The features used in the model were: {0}'.format(missed_star.columns))
    print('Making histogram over these values: {0}'.format(mag_vals))

    # concise axis labels? (assuming exact magnitude use is put in the caption)
    xlabels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4']
    f, axs = plt.subplots(3, 9, figsize=(12,8), sharey=True, sharex=False)

    # plot parameters
    linewidth_hist = 0.5
    hist_ls = 'dotted'
    bins_mag_hist_ugriz = np.linspace(12,28,100)
    bins_mag_hist_w1234 = np.linspace(2,20,100)
    ugriz_lim = [12,28]
    w1234_lim = [2,19]


    ## ------ ------ Precision ------ ------
    # UGRIZ
    for ax, mag_val in zip(axs[0,0:5], mag_vals[0:5]):
        plt.sca(ax)
        # ------ star ------
        x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        p = tp/(tp+fp1+fp2)
        plt.plot(x, p, label='Stars', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
        # log histogram overlaid
        x, s = histvals(df.loc[correct_star.index.values][mag_val], bins=bins_mag_hist_ugriz)
        plt.plot(x, s/(3*s.max()), ls=hist_ls, color=star_c, linewidth=linewidth_hist)

        # ------ galaxy ------
        x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        p = tp/(tp+fp1+fp2)
        plt.plot(x, p, label='Galaxies', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
        # log histogram overlaid
        x, g = histvals(df.loc[correct_galaxy.index.values][mag_val], bins=bins_mag_hist_ugriz)
        plt.plot(x, g/(3*g.max()), ls=hist_ls, color=galaxy_c, linewidth=linewidth_hist)

        # ------ quasar ------
        x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp2 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        p = tp/(tp+fp1+fp2)
        plt.plot(x, p, label='Quasars', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        # log histogram overlaid
        x, q = histvals(df.loc[correct_quasar.index.values][mag_val], bins=bins_mag_hist_ugriz)
        plt.plot(x, q/(3*q.max()), ls=hist_ls, color=quasar_c, linewidth=linewidth_hist)

        if autoscale == False:
            plt.xlim(ugriz_lim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()
        if (mag_val=='psf_u_corr') or (mag_val=='cmod_u_corr'):
            plt.ylabel('Precision') # y axis label for first row on left only

    # WISE
    for ax, mag_val in zip(axs[0,5:9], mag_vals[5:9]):
        plt.sca(ax)
        # ------ star ------
        x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        p = tp/(tp+fp1+fp2)
        plt.plot(x, p, ls=correct_ls, color=star_c, linewidth=linewidth_hist)
        # log histogram overlaid
        x, s = histvals(df.loc[correct_star.index.values][mag_val], bins=bins_mag_hist_w1234)
        plt.plot(x, s/(3*s.max()), ls=hist_ls, color=star_c, linewidth=linewidth_hist)

        # ------ galaxy ------
        x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        p = tp/(tp+fp1+fp2)
        plt.plot(x, p, ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
        # log histogram overlaid
        x, g = histvals(df.loc[correct_galaxy.index.values][mag_val], bins=bins_mag_hist_w1234)
        plt.plot(x, g/(3*g.max()), ls=hist_ls, color=galaxy_c, linewidth=linewidth_hist)

        # ------ quasar ------
        x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp2 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        p = tp/(tp+fp1+fp2)
        plt.plot(x, p, ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        # log histogram overlaid
        x, q = histvals(df.loc[correct_quasar.index.values][mag_val], bins=bins_mag_hist_w1234)
        plt.plot(x, q/(3*q.max()), ls=hist_ls, color=quasar_c, linewidth=linewidth_hist)

        if autoscale == False:
            plt.xlim(w1234_lim)
        if (mag_val=='w4'):
            # dummy plot to get custom legend label:
            plt.plot([0,0], [1,1], label='Histogram per \n class normalised \n to 1/3', ls=hist_ls, color='black', linewidth=linewidth_hist)
            plt.legend(frameon=False, fontsize=4, loc=[0.43,0.21]) # y axis label for first row on left only
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()

    ## ------ ------ Recall ------ ------
    # UGRIZ
    for ax, mag_val in zip(axs[1,0:5], mag_vals[0:5]):
        plt.sca(ax)
        # ------ star ------
        x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        r = tp/(tp+fn1+fn2)
        plt.plot(x, r, label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist)

        # ------ galaxy ------
        x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        r = tp/(tp+fn1+fn2)
        plt.plot(x, r, label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)

        # ------ quasar ------
        x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn2 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        r = tp/(tp+fn1+fn2)
        plt.plot(x, r, label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)

        if autoscale == False:
            plt.xlim(ugriz_lim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()

        if (mag_val=='psf_u_corr') or (mag_val=='cmod_u_corr'):
            plt.ylabel('Recall') # y axis label for middle row, shown on left only
            plt.legend(frameon=False, fontsize=8) # legend box for whole plot middle row left only

    # WISE
    for ax, mag_val in zip(axs[1,5:9], mag_vals[5:9]):
        plt.sca(ax)
        # ------ star ------
        x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        r = tp/(tp+fn1+fn2)
        plt.plot(x, r, label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist)

        # ------ galaxy ------
        x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        r = tp/(tp+fn1+fn2)
        plt.plot(x, r, label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)

        # ------ quasar ------
        x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn2 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        r = tp/(tp+fn1+fn2)
        plt.plot(x, r, label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)

        if autoscale == False:
            plt.xlim(w1234_lim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        ax.minorticks_on()

    ## ------ ------ F1 score ------ ------
    # UGRIZ
    for ax, mag_val in zip(axs[2,0:5], mag_vals[0:5]):
        plt.sca(ax)
        # ------ star ------
        x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # Precision
        x, fp1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # Recall
        x, fn1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # F1
        f1 = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        plt.plot(x, f1, ls=correct_ls, color=star_c, linewidth=linewidth_hist)

        # ------ galaxy ------
        x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # Precision
        x, fp1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # Recall
        x, fn1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # F1
        f1 = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        plt.plot(x, f1, ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)

        # ------ quasar ------
        x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # Precision
        x, fp1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # Recall
        x, fn1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        x, fn2 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
        # F1
        f1 = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        plt.plot(x, f1, ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
        #plt.legend(frameon=False)

        if plot_prob==True:
            # ------------ ------------ PROBABILITIES PER BIN ugriz ------------ ------------
            # get average of probabilities for each bin in the histogram
            g_probs = []
            q_probs = []
            s_probs = []
            g_probs_std = []
            q_probs_std = []
            s_probs_std = []
            # create these copies outside of the for loop below, else this takes order of magnitude longer
            dfg = df.loc[correct_galaxy.index.values]
            dfq = df.loc[correct_quasar.index.values]
            dfs = df.loc[correct_star.index.values]
            x = bins_mag_hist_ugriz # same bins used for f1 score
            # loop over histgoram bins to calculate other stuff (probabilities) per bin. Whilst a bit meh and unpythonic, its quite fast:
            interval = 2 # sample x-axis bins more smoothly?
            for xidx in range(0, len(x[:-1]), interval):
                # note we add 2 to each index because of how histvals function returns same y value for each pair of incremental x values.
                g_probs.append( dfg[ (dfg[mag_val] > x[xidx]) & (dfg[mag_val] < x[xidx+1]) ].prob_g.values.mean() )
                q_probs.append( dfq[ (dfq[mag_val] > x[xidx]) & (dfq[mag_val] < x[xidx+1]) ].prob_q.values.mean() )
                s_probs.append( dfs[ (dfs[mag_val] > x[xidx]) & (dfs[mag_val] < x[xidx+1]) ].prob_s.values.mean() )
                g_probs_std.append( dfg[ (dfg[mag_val] > x[xidx]) & (dfg[mag_val] < x[xidx+1]) ].prob_g.values.std() )
                q_probs_std.append( dfq[ (dfq[mag_val] > x[xidx]) & (dfq[mag_val] < x[xidx+1]) ].prob_q.values.std() )
                s_probs_std.append( dfs[ (dfs[mag_val] > x[xidx]) & (dfs[mag_val] < x[xidx+1]) ].prob_s.values.std() )

            plt.plot([0, 2], [1,1], label='RF probabilities \n for correct classes', color='black', ls=missed_ls, linewidth=linewidth_hist) # dummy plot to get custom legend box for bottom row
            plt.plot(x[0:-1:interval], g_probs, ls=missed_ls, color=galaxy_c, linewidth=linewidth_hist)
            plt.plot(x[0:-1:interval], q_probs, ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
            plt.plot(x[0:-1:interval], s_probs, ls=missed_ls, color=star_c, linewidth=linewidth_hist)
            err_ls='dotted'
            # Dono't show 1-sigma error bars in this plot, they are pretty much constant over the entire x-axis. Perhaps note this in figure caption. These are shown in other plots anyways.
            #plt.plot(x[0:-1:interval], np.array(g_probs)+np.array(g_probs_std), ls=err_ls, color=galaxy_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(g_probs)-np.array(g_probs_std), ls=err_ls, color=galaxy_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(q_probs)+np.array(q_probs_std), ls=err_ls, color=quasar_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(q_probs)-np.array(q_probs_std), ls=err_ls, color=quasar_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(s_probs)+np.array(s_probs_std), ls=err_ls, color=star_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(s_probs)-np.array(s_probs_std), ls=err_ls, color=star_c, linewidth=linewidth_hist)

            # ------------ ------------ ------------ ------------ ------------ ------------

        if autoscale == False:
            plt.xlim(ugriz_lim)
        plt.xlabel(mag_val[4:5]) # replace axis labels. 4:5 removes 'psf_' and '_cor'
        plt.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.minorticks_on()

        if (mag_val=='psf_u_corr') or (mag_val=='cmod_u_corr'):
            plt.ylabel('F1 score')
            plt.legend(frameon=False, fontsize=6, loc='lower right')

    # WISE
    for ax, mag_val in zip(axs[2,5:9], mag_vals[5:9]):
        plt.sca(ax)
        # ------ star ------
        x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # Precision
        x, fp1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # Recall
        x, fn1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # F1
        f1 = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        plt.plot(x, f1, ls=correct_ls, color=star_c, linewidth=linewidth_hist)

        # ------ galaxy ------
        x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # Precision
        x, fp1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # Recall
        x, fn1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # F1
        f1 = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        plt.plot(x, f1, ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)

        # ------ quasar ------
        x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # Precision
        x, fp1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fp2 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # Recall
        x, fn1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        x, fn2 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_w1234, density=density)
        # F1
        f1 = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        plt.plot(x, f1, ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)

        if plot_prob==True:
            # ------------ ------------ PROBABILITIES PER BIN w1234 ------------ ------------
            # get average of probabilities for each bin in the histogram
            g_probs = []
            q_probs = []
            s_probs = []
            g_probs_std = []
            q_probs_std = []
            s_probs_std = []
            # create these copies outside of the for loop below, else this takes order of magnitude longer
            dfg = df.loc[correct_galaxy.index.values]
            dfq = df.loc[correct_quasar.index.values]
            dfs = df.loc[correct_star.index.values]
            # Use same bins
            x = bins_mag_hist_w1234
            # loop over histgoram bins to calculate other stuff (probabilities) per bin. Whilst a bit meh and unpythonic, its quite fast:
            interval = 2 # sample x-axis bins more smoothly?
            for xidx in range(0, len(x[:-1]), interval):
                # note we add 2 to each index because of how histvals function returns same y value for each pair of incremental x values.
                g_probs.append( dfg[ (dfg[mag_val] > x[xidx]) & (dfg[mag_val] < x[xidx+1]) ].prob_g.values.mean() )
                q_probs.append( dfq[ (dfq[mag_val] > x[xidx]) & (dfq[mag_val] < x[xidx+1]) ].prob_q.values.mean() )
                s_probs.append( dfs[ (dfs[mag_val] > x[xidx]) & (dfs[mag_val] < x[xidx+1]) ].prob_s.values.mean() )
                g_probs_std.append( dfg[ (dfg[mag_val] > x[xidx]) & (dfg[mag_val] < x[xidx+1]) ].prob_g.values.std() )
                q_probs_std.append( dfq[ (dfq[mag_val] > x[xidx]) & (dfq[mag_val] < x[xidx+1]) ].prob_q.values.std() )
                s_probs_std.append( dfs[ (dfs[mag_val] > x[xidx]) & (dfs[mag_val] < x[xidx+1]) ].prob_s.values.std() )

            plt.plot(x[0:-1:interval], g_probs, ls=missed_ls, color=galaxy_c, linewidth=linewidth_hist)
            plt.plot(x[0:-1:interval], q_probs, ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
            plt.plot(x[0:-1:interval], s_probs, ls=missed_ls, color=star_c, linewidth=linewidth_hist)
            err_ls='dotted'
            # Dono't show 1-sigma error bars in this plot, they are pretty much constant over the entire x-axis. Perhaps note this in figure caption. These are shown in other plots anyways.
            #plt.plot(x[0:-1:interval], np.array(g_probs)+np.array(g_probs_std), ls=err_ls, color=galaxy_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(g_probs)-np.array(g_probs_std), ls=err_ls, color=galaxy_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(q_probs)+np.array(q_probs_std), ls=err_ls, color=quasar_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(q_probs)-np.array(q_probs_std), ls=err_ls, color=quasar_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(s_probs)+np.array(s_probs_std), ls=err_ls, color=star_c, linewidth=linewidth_hist)
            #plt.plot(x[0:-1:interval], np.array(s_probs)-np.array(s_probs_std), ls=err_ls, color=star_c, linewidth=linewidth_hist)

            # ------------ ------------ ------------ ------------ ------------ ------------

        if autoscale == False:
            plt.xlim(w1234_lim)

        plt.xlabel(mag_val)
        plt.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.minorticks_on()

    f.suptitle(plot_data_label, fontsize=7, y=0.995)
    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    f.savefig('histmatrix-mag-metrics'+plot_data_label+'.pdf', dpi=500)






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Metrics as a function of psudo-magnitude (1-D transform of 10-D feature space)
def plot_metric_curves(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, mag_val='psf_r_corr', plot_data_label='unknown'):
    print('Making metric curve plot...')
    # check some values
    #print(df['feature_1D'].max())
    #print(df['feature_1D'].min())
    #print(df['feature_1D'].mean())

    # hard coded to plot with either of these as x-axis:
    if mag_val=='feature_1D':
        bins_mag_hist_ugriz = np.linspace(-10,28,100) # for 1d feature
        xlabels = '1D feature'
    if mag_val=='psf_r_corr':
        bins_mag_hist_ugriz = np.linspace(8,28,100) # for psf r magnitude
        xlabels = 'PSF r magnitude'
    if mag_val=='psferr_r':
        bins_mag_hist_ugriz = bins_err # for psf r magnitude error
        xlabels = 'PSF r magnitude error'
    if mag_val=='resolvedr':
        bins_mag_hist_ugriz = bins_res # for psf r magnitude error
        xlabels = '| psf_r - cmodel_r | magnitude'

    elinewidth=0.001
    capsize=0.2
    capthick=0.2
    alpha=0.3
    linewidth_hist2=1
    ybottom=-0.05

    zsig=1
    # Even if you train on different features, this function always histograms over the 9 wavebands. Verbal check:
    print('The features used in the model were: {0}'.format(correct_galaxy.columns))
    print('Making histogram over these values: {0}'.format(mag_val))

    # Note, we are searching the original dataframe for matching indices to get magnitudes for the histogram. Whilst this is not needed (and takes longer), it leave this robust for tests such as removing magnitude bands from features, but still creating a histogram over that magnitude band. I think this is worth the extra 30 seconds wait.

    f, axs = plt.subplots(1, 3, figsize=(10,3), sharey=False, sharex=False)

    ## ------ ------ Precision ------ ------
    # UGRIZ
    plt.sca(axs[0])

    # ------ star ------
    x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)

    ps = tp/(tp+fp1+fp2)
    #yerr = 1/np.sqrt(tp+fp1+fp2)
    #perr[perr > 0.25] = 0
    pserr = ( zsig/( (tp+fp1+fp2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2) / (tp+fp1+fp2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, ps, label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
    plt.fill_between(x, ps+pserr, ps-pserr, color=star_c, step='mid', linewidth=0, alpha=alpha)

    # ------ galaxy ------
    x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    pg = tp/(tp+fp1+fp2)
    pgerr = ( zsig/( (tp+fp1+fp2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2) / (tp+fp1+fp2)) + zsig/4 ) ) # wilson score interval
    #yerr=1/np.sqrt(tp+fp1+fp2)
    #yerr[yerr > 0.25] = 0
    plt.plot(x, pg, label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
    plt.fill_between(x, pg+pgerr, pg-pgerr, color=galaxy_c, step='mid', linewidth=0, alpha=alpha)

    # ------ quasar ------
    x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp2 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    pq = tp/(tp+fp1+fp2)
    #yerr=1/np.sqrt(tp+fp1+fp2)
    #yerr[yerr > 0.25] = 0
    pqerr = ( zsig/( (tp+fp1+fp2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2) / (tp+fp1+fp2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, pq, label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
    plt.fill_between(x, pq+pqerr, pq-pqerr, color=quasar_c, step='mid', linewidth=0, alpha=alpha)

    # histogram of each source type
    x, g = histvals_o(df.loc[correct_galaxy.index.values][mag_val], bins=bins_mag_hist_ugriz)
    x, s = histvals_o(df.loc[correct_star.index.values][mag_val], bins=bins_mag_hist_ugriz)
    x, q = histvals_o(df.loc[correct_quasar.index.values][mag_val], bins=bins_mag_hist_ugriz)
    # plot normalised histogram divided by 2 to use up lower half of plot
    # normalise each hist by galaxy count to get relative heights correct
    plt.plot(x, g/(2*g.max()), label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist2)
    plt.plot(x, q/(2*g.max()), label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist2)
    plt.plot(x, s/(2*g.max()), label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist2)

    #axs[0].set_xticklabels([])
    axs[0].minorticks_on()
    plt.ylabel('Precision')
    plt.xlabel(xlabels)
    plt.ylim(bottom=ybottom)
    if (mag_val=='psferr_r') or (mag_val=='resolvedr'):
        plt.xscale('log')
        locmaj = mpl.ticker.LogLocator(base=10,numticks=10) # numticks >> number of expected major ticks
        axs[0].xaxis.set_major_locator(locmaj)
        locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10) # prevent overcrowding
        axs[0].xaxis.set_minor_locator(locmin)
        axs[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())



    ## ------ ------ Recall ------ ------
    # UGRIZ
    plt.sca(axs[1])
    # ------ star ------
    x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    rs = tp/(tp+fn1+fn2)
    #yerr=1/np.sqrt(tp+fn1+fn2)
    #yerr[yerr > 0.25] = 0
    rserr = ( zsig/( (tp+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fn1+fn2) / (tp+fn1+fn2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, rs, label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
    plt.fill_between(x, rs+rserr, rs-rserr, color=star_c, step='mid', linewidth=0, alpha=alpha)

    # ------ galaxy ------
    x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    rg = tp/(tp+fn1+fn2)
    #yerr=1/np.sqrt(tp+fn1+fn2)
    #yerr[yerr > 0.25] = 0
    rgerr = ( zsig/( (tp+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fn1+fn2) / (tp+fn1+fn2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, rg, label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
    plt.fill_between(x, rg+rgerr, rg-rgerr, color=galaxy_c, step='mid', linewidth=0, alpha=alpha)

    # ------ quasar ------
    x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn2 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    rq = tp/(tp+fn1+fn2)
    #yerr=1/np.sqrt(tp+fn1+fn2)
    #yerr[yerr > 0.25] = 0
    rqerr = ( zsig/( (tp+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fn1+fn2) / (tp+fn1+fn2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, rq, label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
    plt.fill_between(x, rq+rqerr, rq-rqerr, color=quasar_c, step='mid', linewidth=0, alpha=alpha)

    # histogram of each source type
    x, g = histvals_o(df.loc[correct_galaxy.index.values][mag_val], bins=bins_mag_hist_ugriz)
    x, s = histvals_o(df.loc[correct_star.index.values][mag_val], bins=bins_mag_hist_ugriz)
    x, q = histvals_o(df.loc[correct_quasar.index.values][mag_val], bins=bins_mag_hist_ugriz)
    # plot normalised histogram divided by 2 to use up lower half of plot
    # normalise each hist by galaxy count to get relative heights correct
    plt.plot(x, g/(2*g.max()), label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist2)
    plt.plot(x, q/(2*g.max()), label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist2)
    plt.plot(x, s/(2*g.max()), label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist2)

    axs[1].minorticks_on()
    #axs[1].set_xticklabels([])
    plt.ylabel('Recall')
    plt.xlabel(xlabels)
    plt.ylim(bottom=ybottom)
    if (mag_val=='psferr_r') or (mag_val=='resolvedr'):
        plt.xscale('log')
        locmaj = mpl.ticker.LogLocator(base=10,numticks=10) # numticks >> number of expected major ticks
        axs[1].xaxis.set_major_locator(locmaj)
        locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10) # prevent overcrowding
        axs[1].xaxis.set_minor_locator(locmin)
        axs[1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


    ## ------ ------ F1 score ------ ------
    # UGRIZ
    plt.sca(axs[2])
    # ------ star ------
    x, tp = histvals_o(df.loc[correct_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # Precision
    x, fp1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # Recall
    x, fn1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn2 = histvals_o(df.loc[missed_star_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # F1
    f1s = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
    #yerr=1/np.sqrt(tp + fp1+fp2 + fn1+fn2)
    #yerr[yerr > 0.25] = 0
    f1serr = ( zsig/( (tp+fp1+fp2+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2+fn1+fn2) / (tp+fp1+fp2+fn1+fn2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, f1s, label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist)
    plt.fill_between(x, f1s+f1serr, f1s-f1serr, color=star_c, step='mid', linewidth=0, alpha=alpha)

    # ------ galaxy ------
    x, tp = histvals_o(df.loc[correct_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # Precision
    x, fp1 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # Recall
    x, fn1 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn2 = histvals_o(df.loc[missed_galaxy_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # F1
    f1g = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
    #yerr=1/np.sqrt(tp + fp1+fp2 + fn1+fn2)
    #yerr[yerr > 0.25] = 0
    f1gerr = ( zsig/( (tp+fp1+fp2+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2+fn1+fn2) / (tp+fp1+fp2+fn1+fn2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, f1g, label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist)
    plt.fill_between(x, f1g+f1gerr, f1g-f1gerr, color=galaxy_c, step='mid', linewidth=0, alpha=alpha)

    # ------ quasar ------
    x, tp = histvals_o(df.loc[correct_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # Precision
    x, fp1 = histvals_o(df.loc[missed_star_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fp2 = histvals_o(df.loc[missed_galaxy_as_quasar.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # Recall
    x, fn1 = histvals_o(df.loc[missed_quasar_as_star.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    x, fn2 = histvals_o(df.loc[missed_quasar_as_galaxy.index.values][mag_val].values, bins=bins_mag_hist_ugriz, density=density)
    # F1
    f1q = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
    #yerr=1/np.sqrt(tp + fp1+fp2 + fn1+fn2)
    #yerr[yerr > 0.25] = 0
    f1qerr = ( zsig/( (tp+fp1+fp2+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2+fn1+fn2) / (tp+fp1+fp2+fn1+fn2)) + zsig/4 ) ) # wilson score interval
    plt.plot(x, f1q, label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist)
    plt.fill_between(x, f1q+f1qerr, f1q-f1qerr, color=quasar_c, step='mid', linewidth=0, alpha=alpha)
    #plt.legend(frameon=False)
    x_tmp = x

    # histogram of each source type
    x, g = histvals_o(df.loc[correct_galaxy.index.values][mag_val], bins=bins_mag_hist_ugriz)
    x, s = histvals_o(df.loc[correct_star.index.values][mag_val], bins=bins_mag_hist_ugriz)
    x, q = histvals_o(df.loc[correct_quasar.index.values][mag_val], bins=bins_mag_hist_ugriz)
    # plot normalised histogram divided by 2 to use up lower half of plot
    # normalise each hist by galaxy count to get relative heights correct
    plt.plot(x, g/(2*g.max()), label='Galaxy', ls=correct_ls, color=galaxy_c, linewidth=linewidth_hist2)
    plt.plot(x, q/(2*g.max()), label='Quasar', ls=correct_ls, color=quasar_c, linewidth=linewidth_hist2)
    plt.plot(x, s/(2*g.max()), label='Star', ls=correct_ls, color=star_c, linewidth=linewidth_hist2)


    # ------------ ------------ PROBABILITIES PER BIN ------------ ------------
    # get average of probabilities for each bin in the histogram
    g_probs = []
    q_probs = []
    s_probs = []
    g_probs_std = []
    q_probs_std = []
    s_probs_std = []
    # create these copies outside of the for loop below, else this takes order of magnitude longer
    dfg = df.loc[correct_galaxy.index.values]
    dfq = df.loc[correct_quasar.index.values]
    dfs = df.loc[correct_star.index.values]
    x = bins_mag_hist_ugriz # same bins used for f1 score
    # loop over histgoram bins to calculate other stuff (probabilities) per bin. Whilst a bit meh and unpythonic, its quite fast:
    interval = 2 # sample x-axis bins more smoothly?
    for xidx in range(0, len(x[:-1]), interval):
        # note we add 2 to each index because of how histvals function returns same y value for each pair of incremental x values.
        g_probs.append( dfg[ (dfg[mag_val] > x[xidx]) & (dfg[mag_val] < x[xidx+1]) ].prob_g.values.mean() )
        q_probs.append( dfq[ (dfq[mag_val] > x[xidx]) & (dfq[mag_val] < x[xidx+1]) ].prob_q.values.mean() )
        s_probs.append( dfs[ (dfs[mag_val] > x[xidx]) & (dfs[mag_val] < x[xidx+1]) ].prob_s.values.mean() )
        g_probs_std.append( dfg[ (dfg[mag_val] > x[xidx]) & (dfg[mag_val] < x[xidx+1]) ].prob_g.values.std() )
        q_probs_std.append( dfq[ (dfq[mag_val] > x[xidx]) & (dfq[mag_val] < x[xidx+1]) ].prob_q.values.std() )
        s_probs_std.append( dfs[ (dfs[mag_val] > x[xidx]) & (dfs[mag_val] < x[xidx+1]) ].prob_s.values.std() )
        # Line below is incredibly slow - do not chain .loc inside conditions - left it here to remind me to be more efficient in the future:
        #g_probs.append( df.loc[correct_galaxy.index.values][ (df.loc[correct_galaxy.index.values][mag_val] > x[xidx]) & (df.loc[correct_galaxy.index.values][mag_val] < x[xidx+2]) ].prob_g.values.mean() )

    plt.plot(x[0:-1:interval], g_probs, ls=missed_ls, color=galaxy_c, linewidth=linewidth_hist)
    plt.plot(x[0:-1:interval], q_probs, ls=missed_ls, color=quasar_c, linewidth=linewidth_hist)
    plt.plot(x[0:-1:interval], s_probs, ls=missed_ls, color=star_c, linewidth=linewidth_hist)
    err_ls='dotted'
    #plt.plot(x[0:-1:interval], np.array(g_probs)+np.array(g_probs_std), ls=err_ls, color=galaxy_c, linewidth=linewidth_hist)
    plt.plot(x[0:-1:interval], np.array(g_probs)-np.array(g_probs_std), ls=err_ls, color=galaxy_c, linewidth=linewidth_hist)
    #plt.plot(x[0:-1:interval], np.array(q_probs)+np.array(q_probs_std), ls=err_ls, color=quasar_c, linewidth=linewidth_hist)
    plt.plot(x[0:-1:interval], np.array(q_probs)-np.array(q_probs_std), ls=err_ls, color=quasar_c, linewidth=linewidth_hist)
    #plt.plot(x[0:-1:interval], np.array(s_probs)+np.array(s_probs_std), ls=err_ls, color=star_c, linewidth=linewidth_hist)
    plt.plot(x[0:-1:interval], np.array(s_probs)-np.array(s_probs_std), ls=err_ls, color=star_c, linewidth=linewidth_hist)

    # too overcrowded with filling between 1sigma error on probabilities
    #plt.fill_between(x[0:-1:interval], np.array(g_probs)+np.array(g_probs_std), np.array(g_probs)-np.array(g_probs_std), facecolor=None, edgecolor=galaxy_c, step='mid', linewidth=0, alpha=0)
    #plt.fill_between(x[0:-1:interval], np.array(q_probs)+np.array(q_probs_std), np.array(q_probs)-np.array(q_probs_std), color=quasar_c, step='mid', linewidth=0, alpha=alpha)
    #plt.fill_between(x[0:-1:interval], np.array(s_probs)+np.array(s_probs_std), np.array(s_probs)-np.array(s_probs_std), color=star_c, step='mid', linewidth=0, alpha=alpha)

    # -------------------------------------------------------------------

    axs[2].minorticks_on()
    plt.ylabel('F1 score')
    plt.xlabel(xlabels)
    plt.ylim(bottom=ybottom)
    if (mag_val=='psferr_r') or (mag_val=='resolvedr'):
        plt.xscale('log')
        locmaj = mpl.ticker.LogLocator(base=10,numticks=10) # numticks >> number of expected major ticks
        axs[2].xaxis.set_major_locator(locmaj)
        locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10) # prevent overcrowding
        axs[2].xaxis.set_minor_locator(locmin)
        axs[2].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    #f.suptitle(plot_data_label, fontsize=7, y=0.995) # if you want the label as a super title above plot
    f.tight_layout()
    #f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    f.savefig('metric-curves'+mag_val+'-'+plot_data_label+'.pdf', dpi=600)

    # save metric scores to disk as dictionary for later use in SDSS_ML_plotmaglim.py
    metrics = {'x':x_tmp, 'g':g, 's':s, 'q':q, 'pg':pg, 'rg':rg, 'f1g':f1g, 'pgerr':pgerr, 'rgerr':rgerr, 'f1gerr':f1gerr, 'pq':pq, 'rq':rq, 'f1q':f1q, 'pqerr':pqerr, 'rqerr':rqerr, 'f1qerr':f1qerr, 'ps':ps, 'rs':rs, 'f1s':f1s, 'pserr':pserr, 'rserr':rserr, 'f1serr':f1serr}
    probs = {'g_probs':g_probs, 's_probs':s_probs, 'q_probs':q_probs}
    metrics_df = pd.DataFrame(metrics)
    probs_df = pd.DataFrame(probs)
    save_obj(metrics_df, 'metrics_df_'+plot_data_label)
    save_obj(probs_df, 'probs_df_'+plot_data_label)






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_probs_hist(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, plot_data_label='unknown'):

    # --- --- --- histgoram of probabilities for correct/missed sources --- ---  ---

    # two plots side by side
    f, axs = plt.subplots(1, 2, figsize=(10,3.5), sharey=False, sharex=False)
    bins_prob = np.linspace(0,1,100)

    plt.sca(axs[0])
    density=False
    cumulative=False
    # correct onjects
    x1, y1 = histvals(df.loc[correct_galaxy.index.values].prob_g.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='correct galaxies', ls=correct_ls, color=galaxy_c)
    x1, y1 = histvals(df.loc[correct_star.index.values].prob_s.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='correct stars', ls=correct_ls, color=star_c)
    x1, y1 = histvals(df.loc[correct_quasar.index.values].prob_q.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='correct quasars', ls=correct_ls, color=quasar_c)

    '''
    # missed objects combined
    x1, y1 = histvals(df.loc[missed_galaxy.index.values].prob_g.values, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed galaxies', ls=missed_ls, color=galaxy_c)
    x1, y1 = histvals(df.loc[missed_star.index.values].prob_s.values, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed stars', ls=missed_ls, color=star_c)
    x1, y1 = histvals(df.loc[missed_quasar.index.values].prob_q.values, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed quasars', ls=missed_ls, color=quasar_c)
    '''

    # missed sources
    x1, y1 = histvals(df.loc[missed_galaxy_as_star.index.values].prob_g.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed galaxies as stars', ls=missed_ls, color=galaxy_c)
    x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values].prob_g.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed galaxies as quasars', ls=missed_ls, color=galaxy_c2)

    x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values].prob_s.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed stars as galaxies', ls=missed_ls, color=star_c)
    x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values].prob_s.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed stars as quasars', ls=missed_ls, color=star_c2)

    x1, y1 = histvals(df.loc[missed_quasar_as_star.index.values].prob_q.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed quasars as stars', ls=missed_ls, color=quasar_c)
    x1, y1 = histvals(df.loc[missed_quasar_as_galaxy.index.values].prob_q.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed quasars as galaxies', ls=missed_ls, color=quasar_c2)

    plt.xlabel('Random forest classification probability')
    plt.ylabel('Number of sources')
    plt.minorticks_on()
    axs[0].tick_params(which='both', right=True)
    plt.yscale('log')
    plt.legend(frameon=False, fontsize=8)


    plt.sca(axs[1])
    density=True
    cumulative=True
    # correct sources
    x1, y1 = histvals(df.loc[correct_galaxy.index.values].prob_g.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='correct galaxies', ls=correct_ls, color=galaxy_c)
    x1, y1 = histvals(df.loc[correct_star.index.values].prob_s.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='correct stars', ls=correct_ls, color=star_c)
    x1, y1 = histvals(df.loc[correct_quasar.index.values].prob_q.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='correct quasars', ls=correct_ls, color=quasar_c)

    # missed sources
    x1, y1 = histvals(df.loc[missed_galaxy_as_star.index.values].prob_g.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed galaxies as stars', ls=missed_ls, color=galaxy_c)
    x1, y1 = histvals(df.loc[missed_galaxy_as_quasar.index.values].prob_g.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed galaxies as quasars', ls=missed_ls, color=galaxy_c2)

    x1, y1 = histvals(df.loc[missed_star_as_galaxy.index.values].prob_s.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed stars as galaxies', ls=missed_ls, color=star_c)
    x1, y1 = histvals(df.loc[missed_star_as_quasar.index.values].prob_s.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed stars as quasars', ls=missed_ls, color=star_c2)

    x1, y1 = histvals(df.loc[missed_quasar_as_star.index.values].prob_q.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed quasars as stars', ls=missed_ls, color=quasar_c)
    x1, y1 = histvals(df.loc[missed_quasar_as_galaxy.index.values].prob_q.values, cumulative=cumulative, bins=bins_prob, density=density)
    plt.plot(x1, y1, label='missed quasars as galaxies', ls=missed_ls, color=quasar_c2)


    plt.xlabel('Random forest classification probability')
    plt.ylabel('Fraction of sources')
    plt.minorticks_on()
    axs[1].tick_params(which='both', right=True)
    #plt.yscale('log')
    #plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('probabilities-hist'+plot_data_label+'.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# custom function to make sequential cmaps from individual colours - some internet person's lovely code:
# http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_probs_hexscatter(df, missed_star, missed_quasar, missed_galaxy, correct_star, correct_quasar, correct_galaxy, missed_star_as_quasar, missed_star_as_galaxy, missed_quasar_as_star, missed_quasar_as_galaxy, missed_galaxy_as_star, missed_galaxy_as_quasar, plot_data_label='unknown'):
    #mpl.rcParams.update({'font.size': 8})
    # --- --- --- scatter plot of all sources with histogram overlaid ---  ---  ---

    # Select plot values:
    val = 'psf_r_corr'
    #val = 'feature_1D'
    # consistent colours for hexbin
    colors_g = [(1,1,1), (112/255,128/255,144/255)]
    cmap_g = make_cmap(colors_g)
    colors_q = [(1,1,1), (255/255,105/255,180/255)]
    cmap_q = make_cmap(colors_q)
    colors_s = [(1,1,1), (30/255,144/255,255/255)]
    cmap_s = make_cmap(colors_s)

    colors_g2 = [(1,1,1), (0/255,0/255,0/255)] # black
    cmap_g2 = make_cmap(colors_g2)
    colors_q2 = [(1,1,1), (0/255,0/255,255/255)] # blue
    cmap_q2 = make_cmap(colors_q2)
    colors_s2 = [(1,1,1), (255/255,165/255,0/255)] # orange
    cmap_s2 = make_cmap(colors_s2)

    gridsize = (60,30) # getting the exact ratio is not trivial... trial and error to get symmetrical hexagons, ensuring binsize is 0.01.
    marker_c = '.'
    marker_m = '.'
    linewidths=0.01 # reduce hexbin linewidths to prevent overlapping
    linewidth=1 # histogram linewidth
    s=0.03 # missed objects scatter plot marker size
    sleg = 13 # legend marker size
    density = False
    f, axs = plt.subplots(1, 3, figsize=(10,4.5), sharey=False, sharex=False)
    #yrange = [-0.02,1.20] # if figsize = 10,5 (to acommodate legend)
    yrange = [-0.02,1.22] # if figsize = 10,4.5

    # correct galaxies
    plt.sca(axs[0])
    bins = np.linspace(15,25,150) # bins for galaxies (diff for each type)
    x1 = df.loc[correct_galaxy.index.values][val].values
    y1 = df.loc[correct_galaxy.index.values].prob_g.values
    #plt.scatter(x1, y1, label='correct galaxies', marker=marker_c, color=galaxy_c, s=s)
    plt.hexbin(x1, y1, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_g)
    im=plt.hexbin(x1, y1, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_g) # for colourbar axs later
    #plt.colorbar(orientation='horizontal', pad=0.01, aspect=40, shrink=1.0, panchor=(0.5, 2.0), anchor=(0.5,1.0))
    x1, y1 = histvals(df.loc[correct_galaxy.index.values].psf_r_corr.values, bins=bins, density=density)
    plt.plot(x1, y1/y1.max(), label='Histogram of correct galaxies', ls=correct_ls, linewidth=linewidth, color=galaxy_c)
    # missed galaxies
    x2 = df.loc[missed_galaxy.index.values][val].values
    y2 = df.loc[missed_galaxy.index.values].prob_g.values
    plt.scatter(x2, y2, marker=marker_m, color=galaxy_c, s=s) # missed galaxies
    #plt.hexbin(x2, y2, gridsize=gridsize, bins='log', linewidths=linewidths, label='missed galaxies', cmap=cmap_g2)
    x2, y2 = histvals(df.loc[missed_galaxy.index.values].psf_r_corr.values, bins=bins, density=density)
    plt.plot(x2, y2/(0.1*y1.max()), label='Histogram of missed galaxies (x10)', ls=missed_ls, linewidth=linewidth, color=galaxy_c)

    plt.xlim(15,25)
    plt.xticks(np.arange(15, 25+1, step=2))
    axs[0].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.ylim(yrange[0], yrange[1])
    plt.yticks(np.arange(0, 1.01, step=0.1))
    axs[0].yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1.01, step=0.02))) # urgh, force them.
    #axs[0].yaxis.set_tick_params(which='minor', left=True)
    plt.scatter(0,0, marker='h', color=galaxy_c, s=sleg, label='Correct galaxies (hexbin colour scale)')
    plt.scatter(0,0, marker='o', facecolors='none', color=galaxy_c, s=sleg, label='Missed galaxies')
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    plt.ylabel('Random forest classification probability')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)
    plt.xlabel('PSF r magnitude')

    ax_divider = make_axes_locatable(axs[0])
    # add an axes above the main axes.
    cax = ax_divider.append_axes("top", size="3%", pad="1%")
    cb = colorbar(im, cax=cax, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps the image
    cax.xaxis.set_ticks_position("top")
    #cax.xaxis.set_tick_params(labelsize='small', labelleft='off')
    cax.axes.get_yaxis().set_visible(False)
    cax.tick_params(labelsize=6)
    labels = cax.get_xticks().tolist()
    labels[0] = '1' # first entry is too small and isn't plotted. force at 1 to prevent log10 inf warnings.
    labels = ['$10^{0:.0f}$'.format(np.log10(int(i))) for i in labels] # convert labels into scientific notation with rasied exponents. ignore first entry as it's not plotted.
    labels[1] = '0' # set second entry to zero (it's a v small value)
    cax.set_xticklabels(labels)
    #cax.ticklabel_format(style='sci')
    #cax.get_yaxis().set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
    #f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    #cax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(f._formatSciNotation('%.0e' % x))))
    plt.minorticks_on()


    # correct quasars
    plt.sca(axs[1])
    bins = np.linspace(15,25,150)
    x1 = df.loc[correct_quasar.index.values][val].values
    y1 = df.loc[correct_quasar.index.values].prob_q.values
    #plt.scatter(x1, y1, label='correct quasars', marker=marker_c, color=quasar_c, s=s)
    plt.hexbin(x1, y1, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_q)
    im = plt.hexbin(x1, y1, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_q) # for cbar
    x1, y1 = histvals(df.loc[correct_quasar.index.values].psf_r_corr.values, bins=bins, density=density)
    plt.plot(x1, y1/y1.max(), label='histogram of correct quasars', ls=correct_ls, linewidth=linewidth, color=quasar_c)
    # missed quasars
    x2 = df.loc[missed_quasar.index.values][val].values
    y2 = df.loc[missed_quasar.index.values].prob_q.values
    plt.scatter(x2, y2, marker=marker_m, color=quasar_c, s=s)
    x2, y2 = histvals(df.loc[missed_quasar.index.values].psf_r_corr.values, bins=bins, density=density)
    plt.plot(x2, y2/y1.max(), label='histogram of missed quasars', ls=missed_ls, linewidth=linewidth, color=quasar_c)
    plt.xlabel('PSF r magnitude')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)

    plt.xlim(15,25)
    plt.xticks(np.arange(15, 25+1, step=2))
    axs[1].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.ylim(yrange[0], yrange[1])
    plt.yticks(np.arange(0, 1.01, step=0.1))
    axs[1].yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1.01, step=0.02)))
    #axs[1].yaxis.set_tick_params(which='minor', left=True)
    plt.scatter(0,0, marker='h', color=quasar_c, s=sleg, label='Correct quasars (hexbin colour scale)')
    plt.scatter(0,0, marker='o', facecolors='none', color=quasar_c, s=sleg, label='Missed quasars')
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)
    plt.xlabel('PSF r magnitude')

    ax_divider = make_axes_locatable(axs[1])
    # add an axes above the main axes.
    cax = ax_divider.append_axes("top", size="3%", pad="1%")
    cb = colorbar(im, cax=cax, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps the image
    cax.xaxis.set_ticks_position("top")
    #cax.xaxis.set_tick_params(labelsize='small', labelleft='off')
    cax.axes.get_yaxis().set_visible(False)
    cax.tick_params(labelsize=6)
    labels = cax.get_xticks().tolist()
    labels[0] = '1' # first entry is too small and isn't plotted. force at 1 to prevent log10 inf warnings.
    labels = ['$10^{0:.0f}$'.format(np.log10(int(i))) for i in labels] # convert labels into scientific notation with rasied exponents. ignore first entry as it's not plotted.
    labels[1] = '0' # set second entry to zero (it's a v small value)
    cax.set_xticklabels(labels)
    plt.minorticks_on()



    # correct stars
    plt.sca(axs[2])
    bins = np.linspace(8,26,150)
    x1 = df.loc[correct_star.index.values][val].values
    y1 = df.loc[correct_star.index.values].prob_s.values
    #plt.scatter(x1, y1, label='correct stars', marker=marker_c, color=star_c, s=s)
    plt.hexbin(x1, y1, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_s)
    im = plt.hexbin(x1, y1, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_s) # for cbar
    x1, y1 = histvals(df.loc[correct_star.index.values].psf_r_corr.values, bins=bins, density=density)
    plt.plot(x1, y1/y1.max(), label='Histogram of correct stars', ls=correct_ls, linewidth=linewidth, color=star_c)
    # missed stars
    x2 = df.loc[missed_star.index.values][val].values
    y2 = df.loc[missed_star.index.values].prob_s.values
    plt.scatter(x2, y2, marker=marker_m, color=star_c, s=s)
    x2, y2 = histvals(df.loc[missed_star.index.values].psf_r_corr.values, bins=bins, density=density)
    plt.plot(x2, y2/y1.max(), label='Histogram of missed stars', ls=missed_ls, linewidth=linewidth, color=star_c)

    plt.xlim(8,26)
    plt.xticks(np.arange(8, 26+1, step=2))
    axs[2].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.ylim(yrange[0], yrange[1])
    plt.yticks(np.arange(0, 1.01, step=0.1))
    axs[2].yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1.01, step=0.02)))
    #axs[2].yaxis.set_tick_params(which='minor', left=True)
    plt.scatter(0,0, marker='h', color=star_c, s=sleg, label='Correct stars (hexbin colour scale)')
    plt.scatter(0,0, marker='o', facecolors='none', color=star_c, s=sleg, label='Missed stars')
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    #plt.ylabel('Random forest classification probability')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)
    plt.xlabel('PSF r magnitude')

    ax_divider = make_axes_locatable(axs[2])
    # add an axes above the main axes.
    cax = ax_divider.append_axes("top", size="3%", pad="1%")
    cb = colorbar(im, cax=cax, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps the image
    cax.xaxis.set_ticks_position("top")
    #cax.xaxis.set_tick_params(labelsize='small', labelleft='off')
    cax.axes.get_yaxis().set_visible(False)
    cax.tick_params(labelsize=6)
    labels = cax.get_xticks().tolist()
    labels[0] = '1' # first entry is too small and isn't plotted. force at 1 to prevent log10 inf warnings.
    labels = ['$10^{0:.0f}$'.format(np.log10(int(i))) for i in labels] # convert labels into scientific notation with rasied exponents. ignore first entry as it's not plotted.
    labels[1] = '0' # set second entry to zero (it's a v small value)
    cax.set_xticklabels(labels)
    plt.minorticks_on()

    #plt.tight_layout() # this doesn't work with the axs divider i used. don't use tight_layout.
    plt.savefig('hexbin-prob-rmag'+plot_data_label+'.pdf', bbox_inches='tight', dpi=700)
    #plt.savefig('prob_1Dfeature_'+plot_data_label+'.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------












#-------------------------------------------------
# Main
#-------------------------------------------------






if __name__ == "__main__": #so you can import this code and run by hand if desired

    # Set some defaults for all plots
    mpl.rcParams.update({'font.size': 8})
    mpl.rcParams.update({'figure.dpi': 400})
    quasar_c = 'hotpink'
    star_c = 'dodgerblue'
    galaxy_c = 'slategrey'
    quasar_c2 = 'black'
    star_c2 = 'blue'
    galaxy_c2 = 'orange'
    correct_ls = '-'
    missed_ls = '--'
    linewidth = 1
    rotation = 0

    ## Paramters that perhaps should be defined inside individual plot functions... for now here they are:
    # Histogram parameters
    density = False
    cumulative = False
    # bins for resolved psf_r - cmodel_r
    bins_res = 10 ** np.linspace(np.log10(1e-5), np.log10(10), 100)
    # bins for magnitude range
    bins_mag = np.linspace(11,24,75)
    # bins for magnitude errors
    bins_err = 10 ** np.linspace(np.log10(1e-4), np.log10(100), 100)
    # parameters for large histogram matrix plots
    linewidth_hist = 0.5
    bins_mag_hist_ugriz = np.linspace(12,28,150)
    bins_mag_hist_w1234 = np.linspace(2,20,150)

    # feature combinations:
    # psf magnitudes feature names
    psf = ['psf_u', 'psf_g', 'psf_r', 'psf_i', 'psf_z']
    # cmodel magnitudes feature names
    cmod = ['cmod_u', 'cmod_g', 'cmod_r', 'cmod_i', 'cmod_z']
    # psf magnitudes corrected for extinction
    psf_ext = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr']
    # cmodel magnitudes corrected for extinction
    cmod_ext = ['cmod_u_corr', 'cmod_g_corr', 'cmod_r_corr', 'cmod_i_corr', 'cmod_z_corr']
    # WISE magnitudes
    wise = ['w1' ,'w2', 'w3', 'w4']
    # All high S/N resolved bands
    resolved_highSN = ['resolvedg','resolvedr', 'resolvedi']
    # errors in r
    errors = ['psferr_r', 'cmoderr_r']
    # Magnitude independent colours
    sdss_colours = ['psf_r_corr_u','psf_r_corr_g','psf_r_corr_i','psf_r_corr_z']
    wise_colours = ['psf_r_corr_w1','psf_r_corr_w2','psf_r_corr_w3','psf_r_corr_w4']
    #feature_columns = sdss_colours + wise_colours + ['resolvedr']

    # set combination of features used from here on:
    feature_columns = psf_ext + wise + ['resolvedr']

    # ------ Load data ------
    print('Loading data...')
    # Load in models and data outputted from the machine learning code (SDSS_ML.py)
    df = load_obj('df')
    pipeline = load_obj('rf_pipeline')
    #data_prep_dict_boss = load_obj('data_prep_dict_boss')
    #data_prep_dict_sdss = load_obj('data_prep_dict_sdss')
    data_prep_dict_all = load_obj('data_prep_dict_all')
    #classes_pred_boss = load_obj('classes_pred_boss')
    #classes_pred_sdss = load_obj('classes_pred_sdss')
    classes_pred_all = load_obj('classes_pred_all')
    classes_pred_all_proba = load_obj('classes_pred_all_proba')

    # Get predicted classes from the RF classifier:
    df_predclass = pd.DataFrame(classes_pred_all, index=data_prep_dict_all['features_test'].index, columns=['class_pred'])
    # Append probabilities to the original df for test data:
    df = df.join(df_predclass, how='left')
    # Get probabilities from the RF classifier:
    df_proba = pd.DataFrame(classes_pred_all_proba, index=data_prep_dict_all['features_test'].index, columns=['prob_g', 'prob_q', 'prob_s'])
    # Append probabilities to the original df for test data:
    df = df.join(df_proba, how='left')



    # Sorting correct and misclassified objects. I know there are a lot of extremely explicit arguments. It means we can produce plots selectively and intuitively.

    # The 'data' argument returned from prepare_misclassified has the predicted classes appended to it. Not required in any further plotting functions, but there just incase it's needed in the future.

    # BOSS
    #data_boss, missed_star_boss, missed_quasar_boss, missed_galaxy_boss, correct_star_boss, correct_quasar_boss, correct_galaxy_boss, missed_star_as_quasar_boss, missed_star_as_galaxy_boss, missed_quasar_as_star_boss, missed_quasar_as_galaxy_boss, missed_galaxy_as_star_boss, missed_galaxy_as_quasar_boss = prepare_classifications(data_prep_dict_boss, classes_pred_boss)

    # SDSS
    #data_sdss, missed_star_sdss, missed_quasar_sdss, missed_galaxy_sdss, correct_star_sdss, correct_quasar_sdss, correct_galaxy_sdss, missed_star_as_quasar_sdss, missed_star_as_galaxy_sdss, missed_quasar_as_star_sdss, missed_quasar_as_galaxy_sdss, missed_galaxy_as_star_sdss, missed_galaxy_as_quasar_sdss = prepare_classifications(data_prep_dict_sdss, classes_pred_sdss)

    # ALL
    data_all, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all = prepare_classifications(data_prep_dict_all, classes_pred_all, ttsplit=True)

    # data_samp, missed_star_samp, missed_quasar_samp, missed_galaxy_samp, correct_star_samp, correct_quasar_samp, correct_galaxy_samp, missed_star_as_quasar_samp, missed_star_as_galaxy_samp, missed_quasar_as_star_samp, missed_quasar_as_galaxy_samp, missed_galaxy_as_star_samp, missed_galaxy_as_quasar_samp = prepare_classifications(data_prep_dict_samp, classes_pred_samp, ttsplit=False)







    # ------ Make plots ------
    # It will take a while to generate all the plots at once.
    # Comment in the one(s) you want to make
    print('Making plots...')

    # plot training vs F1 score. files stored on disk from SDSS_ML.py are given as inputs
    #plot_trainvsf1('train_vs_f1score', 'train_vs_f1score_sampleG', 'train_vs_precision', 'train_vs_precision_sampleG', 'train_vs_recall', 'train_vs_recall_sampleG')

    # plot feature ranking
    # rename features for nice x-axis. Change if using different features
    #feature_labels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4', '$\mathrm{resolved}_\mathrm{r}$']
    #plot_feature_ranking(pipeline, feature_labels)


    # ------ All objects class/instrument histogram ------
    #plot_basic_hists(df)

    #plot_z_hist(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, plot_data_label='')

    # ------ Feature plots ------

    # ALL star/quasar, star/galaxy, quasar/galaxy feature plots for correct and misclassified objects
    #plot_feature_hist(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, plot_data_label='')

    # BOSS: star/quasar, star/galaxy, quasar/galaxy feature plots for correct and misclassified objects
    #plot_feature_hist(df, missed_star_boss, missed_quasar_boss, missed_galaxy_boss, correct_star_boss, correct_quasar_boss, correct_galaxy_boss, missed_star_as_quasar_boss, missed_star_as_galaxy_boss, missed_quasar_as_star_boss, missed_quasar_as_galaxy_boss, missed_galaxy_as_star_boss, missed_galaxy_as_quasar_boss, plot_data_label='BOSS')

    # SDSS: star/quasar, star/galaxy, quasar/galaxy feature plots for correct and misclassified objects
    #plot_feature_hist(df, missed_star_sdss, missed_quasar_sdss, missed_galaxy_sdss, correct_star_sdss, correct_quasar_sdss, correct_galaxy_sdss, missed_star_as_quasar_sdss, missed_star_as_galaxy_sdss, missed_quasar_as_star_sdss, missed_quasar_as_galaxy_sdss, missed_galaxy_as_star_sdss, missed_galaxy_as_quasar_sdss, plot_data_label='SDSS')

    #plot_error_or_resolved_hist(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, val='psferr_r', plot_data_label='')

    #plot_error_or_resolved_hist(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, val='resolvedr', plot_data_label='NoResolvedr')

    # ------ Compare SDSS/BOSS ------

    #plot_compare_sets(df, missed_star_boss, missed_quasar_boss, missed_galaxy_boss, correct_star_boss, correct_quasar_boss, correct_galaxy_boss, missed_star_sdss, missed_quasar_sdss, missed_galaxy_sdss, correct_star_sdss, correct_quasar_sdss, correct_galaxy_sdss, plot_data_label='SDSSBOSS')

    # ------ Histogram feature matrix ------

    # These take a bit longer to run because we are searching through the original df to grab features, just in case we decide to remove features from the training but still include them in these histogram plots.

    # ALL
    #plot_histogram_matrix(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, plot_data_label='')

    # BOSS
    #plot_histogram_matrix(df, missed_star_boss, missed_quasar_boss, missed_galaxy_boss, correct_star_boss, correct_quasar_boss, correct_galaxy_boss, missed_star_as_quasar_boss, missed_star_as_galaxy_boss, missed_quasar_as_star_boss, missed_quasar_as_galaxy_boss, missed_galaxy_as_star_boss, missed_galaxy_as_quasar_boss, plot_data_label='BOSS')

    # SDSS
    #plot_histogram_matrix(df, missed_star_sdss, missed_quasar_sdss, missed_galaxy_sdss, correct_star_sdss, correct_quasar_sdss, correct_galaxy_sdss, missed_star_as_quasar_sdss, missed_star_as_galaxy_sdss, missed_quasar_as_star_sdss, missed_quasar_as_galaxy_sdss, missed_galaxy_as_star_sdss, missed_galaxy_as_quasar_sdss, plot_data_label='SDSS')

    # ------ Histogram feature matrix of precision recall f1 probabilities ------
    # saved as: hist-magfeatures-metrics-*.pdf
    # ALL
    #plot_histogram_matrix_f1(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, plot_data_label='', autoscale=False, plot_prob=True)

    # BOSS
    #plot_histogram_matrix_f1(df, missed_star_boss, missed_quasar_boss, missed_galaxy_boss, correct_star_boss, correct_quasar_boss, correct_galaxy_boss, missed_star_as_quasar_boss, missed_star_as_galaxy_boss, missed_quasar_as_star_boss, missed_quasar_as_galaxy_boss, missed_galaxy_as_star_boss, missed_galaxy_as_quasar_boss, plot_data_label='BOSS', autoscale=False, plot_prob=True)

    # SDSS
    #plot_histogram_matrix_f1(df, missed_star_sdss, missed_quasar_sdss, missed_galaxy_sdss, correct_star_sdss, correct_quasar_sdss, correct_galaxy_sdss, missed_star_as_quasar_sdss, missed_star_as_galaxy_sdss, missed_quasar_as_star_sdss, missed_quasar_as_galaxy_sdss, missed_galaxy_as_star_sdss, missed_galaxy_as_quasar_sdss, plot_data_label='SDSS', autoscale=False, plot_prob=True)


    # ------ Metric curves - Histogram precision recall f1 1D feature ------
    # to investigate magnitude limit and make plots, run this function a few times
    '''
    plot_metric_curves(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, mag_val='psf_r_corr', plot_data_label='')

    plot_metric_curves(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, mag_val='feature_1D', plot_data_label='')

    plot_metric_curves(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, mag_val='psferr_r', plot_data_label='')

    plot_metric_curves(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, mag_val='resolvedr', plot_data_label='')
    '''

    # ------ Probability of classification per class - Hexbin scatter hist ------

    #plot_probs_hist(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, plot_data_label='')

    #plot_probs_hexscatter(df, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all, plot_data_label='')


    # End
