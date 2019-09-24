# Written by Alex Clarke - https://github.com/informationcake/SDSS-ML
# It will classify new objects from their features, and generate plots.

# Pre-requisits: Run SDSS_ML.py to obtain the following .pkl model:
# df.pkl
# rf_pipeline.pkl
# You should understand this model via output plots made using SDSS_ML_analysis.py

# Version numbers
# python: 3.6.1
# pandas: 0.25.0
# numpy: 1.17.0
# scipy: 1.3.1
# matplotlib: 3.1.1
# sklearn: 0.20.0
# datashader: 0.7.0

import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg',warn=False, force=True) #set MPL backend.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import pickle #save/load python data objects (dictionaries/arrays)
import multiprocessing
import itertools
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import datetime
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

# Previous functions from SDSS_ML.py and SDSS_ML_analysis.py
from SDSS_ML_analysis import save_obj, load_obj, histvals, make_cmap

# functions:
# drop_duplicates
# drop_duplicates_function
# classify_new_sources_batched
# plot_new_hist_batched
# classify_new_sources
# plot_new_hist
# plot_new_maghist
# plot_new_hexbin
# plot_new_feature_hist
# plot_newsources_redshift
# load_spec
# print_result_numbers
# save_catalogue_todisk



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # different to SDSS_ML.py drop_duplicates because using objid instead of specObjID
def drop_duplicates(df):
    # Find duplicate SpecOBJIDs:
    df_dup = df[df.objid.duplicated()==True]
    i = df[df.objid.duplicated()==True].objid
    drop_idxs = []
    # Loop over them, work out maximum match distance, append to list.
    for idx in i:
        m = df[df.objid==idx].match_dist
        drop_idxs.append(m.idxmax())
    # Drop entires with maximum match_dist, leaving best matching object.
    df = df.drop(drop_idxs)
    return df






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# drop duplicates as separate function because it takes hours
def drop_duplicates_function(datafile):
    print(' Dropping duplicates, takes 2 hours and requires ~50 GB of RAM...')
    print(' Reading in file: {0}'.format(datetime.datetime.utcnow()))
    df = pd.read_csv(datafile)
    print(' There are {0} rows. \n Dropping duplicate WISE matches...'.format(len(df)))
    df = drop_duplicates(df)
    print(' There are now {0} rows. {1}'.format(len(df), datetime.datetime.utcnow()))
    save_obj(df, datafile+'_DropDuplicates')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # Classify new sources in chunks, return and save to disk dfs of class labels and probabilities
def classify_new_sources_batched(datafile, rf_pipeline, feature_columns, nrows=100000, chunksize=1000):
    print('Classifying {0} rows from {1}, processing {2} at a time.'.format(nrows, datafile, chunksize))
    # initialise empty arrays to append each chunk of data to
    new_classes_all = np.empty((0))
    new_classes_proba_all = np.empty((0,3)) # probabilities are formatted: [[pg, pq, ps],[pg, pq, ps],...]
    # loop through chunks. np.ceil return float, hence int(). i is just for printing.
    for i, df_chunk in zip(range(0, int(np.ceil(nrows/chunksize)), 1), pd.read_csv(datafile, chunksize=chunksize, nrows=nrows)):
        print('Classifying chunk {0}/{1} ...'.format(i+1, int(np.ceil(nrows/chunksize))))
        # calculate resolved feature for chunk
        df_chunk['resolvedr'] = np.sqrt((df_chunk.psfmag_r - df_chunk.cmod_r)**2)
        # predict classes and probabilities
        new_classes = rf_pipeline.predict(df_chunk[feature_columns])
        new_classes_proba = rf_pipeline.predict_proba(df_chunk[feature_columns])
        # append chunk results to master arrays
        new_classes_all = np.append(new_classes_all, new_classes)
        new_classes_proba_all = np.append(new_classes_proba_all, new_classes_proba, axis=0)
    # Turn master result arrays into dfs
    df_new_classes_all = pd.DataFrame( new_classes_all, columns=['class_pred'] )
    df_new_classes_proba_all = pd.DataFrame( new_classes_proba_all, columns=['prob_g', 'prob_q', 'prob_s'] )
    # Save df to disk
    save_obj(df_new_classes_all, 'new_classes_all')
    save_obj(df_new_classes_proba_all, 'new_classes_proba_all')

    return df_new_classes_all, df_new_classes_proba_all






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # plot histogram of new classes
def plot_new_hist_batched(datafile, df_classes, df_proba, feature_columns, nrows=10000, chunksize=10000):
    print('Loading data for plotting in batches...')
    bins_r = np.linspace(12,28,150)
    bins_p = np.linspace(0,1,100)
    #x1, y1 = histvals(df_new_sources.psfmag_r, cumulative=cumulative, bins=bins, density=density)
    #plt.plot(x1[x1<0.6], y1[x1<0.6], label='missed quasars as stars', ls=missed_ls, color=quasar_c)
    yg = np.zeros(len(bins_r)*2) # Initialise empty array for histogram values. x2 beacuse of how histvals works.
    yq = np.zeros(len(bins_r)*2) # Quasar
    ys = np.zeros(len(bins_r)*2) # Stars
    yg_prob = np.zeros(len(bins_p)*2) # Initialise empty array for histogram values. x2 beacuse of how histvals works.
    yq_prob = np.zeros(len(bins_p)*2) # Quasar
    ys_prob = np.zeros(len(bins_p)*2) # Stars
    # loop through chunks. np.ceil return float, hence int().
    for i, df_chunk in zip( range(0, int(np.ceil(nrows/chunksize)), 1), pd.read_csv(datafile, chunksize=chunksize, nrows=nrows) ):
        print('Sorting classes for histogram: chunk {0}/{1} ...'.format(i+1, int(np.ceil(nrows/chunksize))))

        # histogram r magnitude per class:
        x, yg_chunk = histvals(df_chunk.loc[ df_classes[df_classes['class_pred']=='GALAXY'].index.intersection(df_chunk.index) ].psfmag_r, cumulative=cumulative, bins=bins_r, density=density)
        x, yq_chunk = histvals(df_chunk.loc[ df_classes[df_classes['class_pred']=='QSO'].index.intersection(df_chunk.index) ].psfmag_r, cumulative=cumulative, bins=bins_r, density=density)
        x, ys_chunk = histvals(df_chunk.loc[ df_classes[df_classes['class_pred']=='STAR'].index.intersection(df_chunk.index) ].psfmag_r, cumulative=cumulative, bins=bins_r, density=density)
        yg = yg + np.array(yg_chunk)
        yq = yq + np.array(yq_chunk)
        ys = ys + np.array(ys_chunk)

        # histogram probabilities per class:
        xp, yg_prob_chunk = histvals(df_proba.loc[ df_classes[df_classes['class_pred']=='GALAXY'].index.intersection(df_chunk.index) ].prob_g, cumulative=cumulative, bins=bins_p, density=density)
        xp, yq_prob_chunk = histvals(df_proba.loc[ df_classes[df_classes['class_pred']=='QSO'].index.intersection(df_chunk.index) ].prob_q, cumulative=cumulative, bins=bins_p, density=density)
        xp, ys_prob_chunk = histvals(df_proba.loc[ df_classes[df_classes['class_pred']=='STAR'].index.intersection(df_chunk.index) ].prob_s, cumulative=cumulative, bins=bins_p, density=density)
        yg_prob = yg_prob + np.array(yg_prob_chunk)
        yq_prob = yq_prob + np.array(yq_prob_chunk)
        ys_prob = ys_prob + np.array(ys_prob_chunk)

    print('plotting...')
    plt.plot(x, yg, label='Galaxies', color=galaxy_c, linewidth=linewidth)
    plt.plot(x, yq, label='Quasars', color=quasar_c, linewidth=linewidth)
    plt.plot(x, ys, label='Stars', color=star_c, linewidth=linewidth)
    # sum of all:
    plt.plot(x, yg+yq+ys, label='All', color='black', linewidth=0.3, ls='--')

    plt.xlabel('PSF r magnitude')
    plt.ylabel('Number of sources')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('New-sources-cum-hist-psfr.pdf')
    plt.clf()

    # plot probabilities
    plt.plot(xp, yg_prob, label='Galaxies', color=galaxy_c, linewidth=linewidth)
    plt.plot(xp, yq_prob, label='Quasars', color=quasar_c, linewidth=linewidth)
    plt.plot(xp, ys_prob, label='Stars', color=star_c, linewidth=linewidth)
    # sum of all:
    plt.plot(xp, yg_prob+yq_prob+ys_prob, label='All', color='black', linewidth=0.3, ls='--')

    plt.xlabel('RF classification probability')
    plt.ylabel('Number of sources')
    #plt.yscale('log')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('New-sources-cum-hist-prob.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------







def classify_new_sources(datafile, feature_columns):
    print(' Classifying all rows from {0} at once. Takes 15 minutes, but requires ~50 GB of RAM.'.format(datafile))
    rf_pipeline = load_obj('rf_pipeline')
    print(' Reading in file: {0} - {1}'.format(datafile, datetime.datetime.utcnow()))
    df = load_obj(datafile+'_DropDuplicates')
    # reset index of the df because we drops some rows, and want index to not have gaps.
    df = df.reset_index(drop=True)
    print(' There are {0} rows'.format(len(df)))
    print(' Calculating resolved feature for all rows - {0}'.format(datetime.datetime.utcnow()))
    df['resolvedr'] = np.sqrt((df.psf_r_corr - df.cmod_r_corr)**2)
    print(' Predicting classes for all rows - {0}'.format(datetime.datetime.utcnow()))
    classes = rf_pipeline.predict(df[feature_columns])
    probabilities = rf_pipeline.predict_proba(df[feature_columns])
    print(' Turning arrays into DataFrames - {0}'.format(datetime.datetime.utcnow()))
    df_classes = pd.DataFrame( classes, columns=['class_pred'] ) # Turn arrays into dfs
    df_probabilities = pd.DataFrame( probabilities, columns=['prob_g', 'prob_q', 'prob_s'] )
    print(' Appending class/probability dfs to original df - {0}'.format(datetime.datetime.utcnow()))
    df = df.join(df_classes, how='left')
    df = df.join(df_probabilities, how='left')
    print(' Saving new df to disk... {0}'.format(datetime.datetime.utcnow()))
    save_obj(df, datafile+'_classified')
    print(' Done! {0}'.format(datetime.datetime.utcnow()))






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------





    # histograms
def plot_new_hist(df, df_g, df_q, df_s):
    print('Plotting histograms... New-sources-hist * .pdf')
    bins_r = np.linspace(0,27,150)
    bins_p = np.linspace(0,1,100)

    # plot histogram over psf_r magnitude
    xrg, yg = histvals(df_g.psf_r_corr, bins=bins_r, cumulative=False, density=False)
    xrq, yq = histvals(df_q.psf_r_corr, bins=bins_r, cumulative=False, density=False)
    xrs, ys = histvals(df_s.psf_r_corr, bins=bins_r, cumulative=False, density=False)
    fig = plt.figure()
    plt.plot(xrg, yg, label='Predicted galaxies', color=galaxy_c, linewidth=linewidth)
    plt.plot(xrq, yq, label='Predicted quasars', color=quasar_c, linewidth=linewidth)
    plt.plot(xrs, ys, label='Predicted stars', color=star_c, linewidth=linewidth)
    # sum of all:
    plt.plot(xrg, yg+yq+ys, label='All photometric sources', color='black', linewidth=0.2, ls='--')

    # Spectra sources
    x1, y1 = histvals(df[df['class']=='GALAXY'].psf_r_corr, bins=bins_r, cumulative=False, density=False)
    plt.plot(x1, y1, label='Galaxies with spectra', ls='--', linewidth=0.5, color=galaxy_c)
    x1, y1 = histvals(df[df['class']=='QSO'].psf_r_corr, bins=bins_r, cumulative=False, density=False)
    plt.plot(x1, y1, label='Quasars with spectra', ls='--', linewidth=0.5, color=quasar_c)
    x1, y1 = histvals(df[df['class']=='STAR'].psf_r_corr, bins=bins_r, cumulative=False, density=False)
    plt.plot(x1, y1, label='Stars with spectra', ls='--', linewidth=0.5, color=star_c)

    plt.xticks(np.arange(min(bins_r), max(bins_r)+1, step=2))
    plt.gca().xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.xlim(0,27)
    plt.xlabel('PSF r magnitude')
    plt.ylabel('Number of sources')
    plt.yscale('log')
    # ticks on both sides of plot
    plt.tick_params(which='both', right=True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('New-sources-hist-psfr.pdf')
    plt.clf()



    # Plot histogram over probabilities
    f, axs = plt.subplots(1, 2, figsize=(10,3.5), sharey=False, sharex=False) # two plots side by side
    plt.sca(axs[0])
    xpg, yg_prob = histvals(df_g.prob_g, bins=bins_p, cumulative=False, density=False)
    xpq, yq_prob = histvals(df_q.prob_q, bins=bins_p, cumulative=False, density=False)
    xps, ys_prob = histvals(df_s.prob_s, bins=bins_p, cumulative=False, density=False)
    plt.plot(xpg, yg_prob, label='Predicted galaxies', color=galaxy_c, linewidth=linewidth)
    plt.plot(xpq, yq_prob, label='Predicted quasars', color=quasar_c, linewidth=linewidth)
    plt.plot(xps, ys_prob, label='Predicted stars', color=star_c, linewidth=linewidth)
    #plt.plot(xpg, yg_prob+yq_prob+ys_prob, label='All', color='black', linewidth=0.2, ls='--') # sum of all

    # get correct objects from spec df to overlay on plot
    correct_galaxy = df[ (df['class'] == df['class_pred']) & (df['class'] == 'GALAXY') ]
    correct_quasar = df[ (df['class'] == df['class_pred']) & (df['class'] == 'QSO') ]
    correct_star = df[ (df['class'] == df['class_pred']) & (df['class'] == 'STAR') ]
    # Plot spectrosopically observed correct sources
    x1, y1 = histvals(df.loc[correct_galaxy.index].prob_g, bins=bins_p, cumulative=False, density=False)
    plt.plot(x1, y1, label='Correct galaxies with spectra', ls='--', linewidth=0.5, color=galaxy_c)
    x1, y1 = histvals(df.loc[correct_quasar.index].prob_q, bins=bins_p, cumulative=False, density=False)
    plt.plot(x1, y1, label='Correct quasars with spectra', ls='--', linewidth=0.5, color=quasar_c)
    x1, y1 = histvals(df.loc[correct_star.index].prob_s, bins=bins_p, cumulative=False, density=False)
    plt.plot(x1, y1, label='Correct stars with spectra', ls='--', linewidth=0.5, color=star_c)
    plt.xlabel('RF classification probability')
    plt.ylabel('Number of sources')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.minorticks_on()
    axs[0].tick_params(which='both', right=True)

    # plot cumulative normalised histogram over probabilities
    plt.sca(axs[1])
    print(bins_p)
    xpg_cum, yg_prob_cum = histvals(df_g.prob_g, bins=bins_p, cumulative=True, density=True)
    xpq_cum, yq_prob_cum = histvals(df_q.prob_q, bins=bins_p, cumulative=True, density=True)
    xps_cum, ys_prob_cum = histvals(df_s.prob_s, bins=bins_p, cumulative=True, density=True)
    plt.plot(xpg_cum, yg_prob_cum, label='Predicted galaxies', color=galaxy_c, linewidth=linewidth)
    plt.plot(xpq_cum, yq_prob_cum, label='Predicted quasars', color=quasar_c, linewidth=linewidth)
    plt.plot(xps_cum, ys_prob_cum, label='Predicted stars', color=star_c, linewidth=linewidth)
    print(xpg_cum)
    print( len(bins_p), len(xpg_cum) )
    # Plot spectrosopically observed correct sources
    x1, y1 = histvals(df.loc[correct_galaxy.index].prob_g, bins=bins_p, cumulative=True, density=True)
    plt.plot(x1, y1, label='Correct galaxies with spectra', ls='--', linewidth=0.5, color=galaxy_c)
    x1, y1 = histvals(df.loc[correct_quasar.index].prob_q, bins=bins_p, cumulative=True, density=True)
    plt.plot(x1, y1, label='Correct quasars with spectra', ls='--', linewidth=0.5, color=quasar_c)
    x1, y1 = histvals(df.loc[correct_star.index].prob_s, bins=bins_p, cumulative=True, density=True)
    plt.plot(x1, y1, label='Correct stars with spectra', ls='--', linewidth=0.5, color=star_c)
    plt.xlabel('RF classification probability')
    plt.ylabel('Fraction of sources per class')
    plt.legend(frameon=False)
    plt.minorticks_on()
    axs[1].tick_params(which='both', right=True)
    plt.tight_layout()
    plt.savefig('New-sources-hist-prob.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_new_maghist(df_g, df_q, df_s):
    print('Plotting histogram over all UGRIZ and W1234 bands... New-sources-maghist.pdf')
    # histogram over these values:
    bins_ugriz = np.linspace(0,35,200)
    bins_w1234 = np.linspace(0,20,200)
    ugriz_lim = [0,35]
    w1234_lim = [0,20]
    #all_ylim = [0,1e7]
    ls = '-'
    linewidth_hist = 0.5
    # define labels for the x axis
    xlabels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4'] # in case we want more concise labels.
    # Only keep magnitude features to plot (remove resolvedr feature):
    mag_vals = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4']
    f, axs = plt.subplots(1, 9, figsize=(16,4), sharey=True, sharex=False)

    # UGRIZ
    for ax, mag_val, xlab in zip(axs[0:5], mag_vals[0:5], xlabels[0:5]):
        plt.sca(ax)
        x1, y1 = histvals(df_g[mag_val].values, bins=bins_ugriz)
        plt.plot(x1, y1, label='Predicted galaxies', ls=ls, color=galaxy_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df_q[mag_val].values, bins=bins_ugriz)
        plt.plot(x2, y2, label='Predicted quasars', ls=ls, color=quasar_c, linewidth=linewidth_hist)
        x3, y3 = histvals(df_s[mag_val].values, bins=bins_ugriz)
        plt.plot(x3, y3, label='Predicted stars', ls=ls, color=star_c, linewidth=linewidth_hist)
        plt.yscale('log')
        plt.xlim(ugriz_lim)
        #plt.ylim(all_ylim)
        #ax.set_xticklabels([])
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
        if mag_val=='psfmag_u':
            plt.ylabel('Number of sources')
            plt.legend(frameon=False, fontsize=6, loc='upper left')
        plt.xlabel(xlab) # replace axis labels. 4:5 removes 'psf_' and '_cor'
        plt.tick_params(axis='x', which='both', bottom=True, top=True)
        # ticks on both sides of plot
        plt.tick_params(axis='y', which='both', right=True)
        ax.minorticks_on()

    # WISE
    for ax, mag_val, xlab in zip(axs[5:9], mag_vals[5:9], xlabels[5:9]):
        plt.sca(ax)
        x1, y1 = histvals(df_g[mag_val].values, bins=bins_w1234)
        plt.plot(x1, y1, label='Predicted galaxies', ls=ls, color=galaxy_c, linewidth=linewidth_hist)
        x2, y2 = histvals(df_q[mag_val].values, bins=bins_w1234)
        plt.plot(x2, y2, label='Predicted quasars', ls=ls, color=quasar_c, linewidth=linewidth_hist)
        x3, y3 = histvals(df_s[mag_val].values, bins=bins_w1234)
        plt.plot(x3, y3, label='Predicted stars', ls=ls, color=star_c, linewidth=linewidth_hist)
        plt.yscale('log')
        plt.xlim(w1234_lim)
        #plt.ylim(all_ylim)
        #ax.set_xticklabels([])
        #plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        plt.xlabel(xlab)
        plt.tick_params(axis='x', which='both', bottom=True, top=True)
        # ticks on both sides of plot
        plt.tick_params(axis='y', which='both', right=True)
        ax.minorticks_on()

    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    f.savefig('New-sources-maghist.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_new_hexbin(df_g, df_q, df_s):
    # --- --- --- scatter plot of all sources with histogram overlaid ---  ---  ---
    print('Plotting... New-sources-prob-hexbin.pdf')
    mpl.rcParams.update({'font.size': 8})
    # Select plot values:
    val = 'psfmag_r'
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

    gridsize = (60,30)
    marker = '.'
    linewidths = 0.01 # reduce hexbin linewidths to prevent overlapping
    linewidth = 1 # histogram linewidth
    s = 0.03 # missed objects scatter plot marker size
    sleg = 13 # legend marker size
    density = False
    yrange = [-0.02,1.11]

    f, axs = plt.subplots(1, 3, figsize=(10,4.5), sharey=False, sharex=False)

    # galaxies
    plt.sca(axs[0])
    bins = np.linspace(10,27,150) # bins for galaxies (diff for each type)
    #plt.scatter(df[df.class_pred == 'GALAXY'].psfmag_r, df[df.class_pred == 'GALAXY'].prob_g, label='Predicted galaxies', color=galaxy_c, s=s)
    plt.hexbin(df_g.psf_r_corr, df_g.prob_g, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_g)
    im = plt.hexbin(df_g.psf_r_corr, df_g.prob_g, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_g)
    x1, y1 = histvals(df_g.psf_r_corr.values, bins=bins, density=density)
    plt.plot(x1, y1/y1.max(), label='Histogram of predicted galaxies', linewidth=linewidth, color=galaxy_c)
    plt.xlim(min(bins),max(bins))
    plt.xticks(np.arange(min(bins), max(bins)+1, step=2))
    axs[0].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.ylim(yrange[0], yrange[1])
    plt.yticks(np.arange(0, 1.01, step=0.1))
    axs[0].yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1.01, step=0.02)))
    #axs[0].yaxis.set_tick_params(which='minor', left=True)
    plt.scatter(0,0, marker='h', color=galaxy_c, s=sleg, label='Predicted galaxies (hexbin colour scale)')
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    plt.ylabel('Random forest classification probability')
    plt.xlabel('PSF r magnitude')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)

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
    plt.minorticks_on()

    # quasars
    plt.sca(axs[1])
    bins = np.linspace(2,25,150)
    #plt.scatter(df[df.class_pred == 'QSO'].psfmag_r, df[df.class_pred == 'QSO'].prob_q, label='Predicted quasars', color=quasar_c, s=s)
    plt.hexbin(df_q.psf_r_corr, df_q.prob_q, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_q)
    im = plt.hexbin(df_q.psf_r_corr, df_q.prob_q, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_q)
    x1, y1 = histvals(df_q.psf_r_corr.values, bins=bins, density=density)
    plt.plot(x1, y1/y1.max(), label='Histogram of predicted quasars', linewidth=linewidth, color=quasar_c)
    plt.xlim(min(bins),max(bins))
    plt.xticks(np.arange(min(bins), max(bins)+1, step=2))
    axs[1].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.ylim(yrange[0], yrange[1])
    plt.yticks(np.arange(0, 1.01, step=0.1))
    axs[1].yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1.01, step=0.02)))
    #axs[1].yaxis.set_tick_params(which='minor', left=True)
    plt.scatter(0,0, marker='h', color=quasar_c, s=sleg, label='Predicted quasars (hexbin colour scale)')
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    plt.xlabel('PSF r magnitude')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)

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


    # stars
    plt.sca(axs[2])
    bins = np.linspace(2,27,150)
    #plt.scatter(df[df.class_pred == 'STAR'].psfmag_r, df[df.class_pred == 'STAR'].prob_s, label='Predicted stars', color=star_c, s=s)
    plt.hexbin(df_s.psf_r_corr, df_s.prob_s, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_s)
    im = plt.hexbin(df_s.psf_r_corr, df_s.prob_s, gridsize=gridsize, bins='log', linewidths=linewidths, cmap=cmap_s)
    x1, y1 = histvals(df_s.psf_r_corr.values, bins=bins, density=density)
    plt.plot(x1, y1/y1.max(), label='Histogram of predicted stars', linewidth=linewidth, color=star_c)
    plt.xlim(min(bins),max(bins))
    plt.xticks(np.arange(min(bins), max(bins)+1, step=2))
    axs[2].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    plt.ylim(yrange[0], yrange[1])
    plt.yticks(np.arange(0, 1.01, step=0.1))
    axs[2].yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1.01, step=0.02)))
    #axs[2].yaxis.set_tick_params(which='minor', left=True)
    plt.xlabel('PSF r magnitude')
    #plt.xlabel('1D feature')
    #plt.ylabel('Random forest classification probability')
    #plt.yscale('log')
    plt.scatter(0,0, marker='h', color=star_c, s=sleg, label='Predicted stars (hexbin colour scale)')
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)
    plt.minorticks_on()

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

    plt.savefig('New-sources-prob-hexbin.pdf', bbox_inches='tight', dpi=700)







    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_new_feature_hist(df_spec, df_g, df_q, df_s):
    print('Plotting... New-sources-features-perclass.pdf')
    spec_capsize = 1
    photo_capsize = 4
    linewidth = 1.2
    linewidth_s = 0.8
    elinewidth = 0.5
    capthick = 0.5
    # bins for resolved psf_r - cmodel_r
    bins_res = 10 ** np.linspace(np.log10(1e-5), np.log10(10), 100)
    # define labels for the x axis
    xlabels = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4'] # more concise labels.
    # Only keep magnitude features to plot (remove resolvedr feature):
    plot_columns = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4']
    # OLD # plot_columns = ['psfmag_u', 'psfmag_g', 'psfmag_r', 'psfmag_i', 'psfmag_z', 'w1', 'w2', 'w3', 'w4']

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.sca(ax1) # plt. gives easier access to some parms. Also means pandas.plot works nicely for current axis.
    # set up transform to offset each of the lines so they dono't overlap
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData

    # Photometric sources
    plt.errorbar(plot_columns, df_g[plot_columns].mean(), yerr=df_g[plot_columns].std(), color=galaxy_c, ls=ls, capsize=photo_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='Predicted galaxies: {0}'.format(len(df_g)), transform=trans+offset(-5))
    plt.errorbar(plot_columns, df_q[plot_columns].mean(), yerr=df_q[plot_columns].std(), color=quasar_c, ls=ls, capsize=photo_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='Predicted quasars: {0}'.format(len(df_q)), transform=trans+offset(-1))
    plt.errorbar(plot_columns, df_s[plot_columns].mean(), yerr=df_s[plot_columns].std(), color=star_c, ls=ls, capsize=photo_capsize, capthick=capthick, linewidth=linewidth, elinewidth=elinewidth, label='Predicted stars: {0}'.format(len(df_s)), transform=trans+offset(+3))

    # Spectroscopic sources
    a = plt.errorbar(plot_columns, df_spec[df_spec['class']=='GALAXY'][plot_columns].mean(), yerr=df_spec[df_spec['class']=='GALAXY'][plot_columns].std(), color=galaxy_c, ls='--', capsize=spec_capsize, capthick=capthick, linewidth=linewidth_s, elinewidth=elinewidth, label='Galaxies with spectra: {0}'.format(len(df_spec[df_spec['class']=='GALAXY'])), transform=trans+offset(-3))
    a[-1][0].set_linestyle('--')
    b = plt.errorbar(plot_columns, df_spec[df_spec['class']=='QSO'][plot_columns].mean(), yerr=df_spec[df_spec['class']=='QSO'][plot_columns].std(), color=quasar_c, ls='--', capsize=spec_capsize, capthick=capthick, linewidth=linewidth_s, elinewidth=elinewidth, label='Quasars with spectra: {0}'.format(len(df_spec[df_spec['class']=='QSO'])), transform=trans+offset(+1))
    b[-1][0].set_linestyle('--')
    c = plt.errorbar(plot_columns, df_spec[df_spec['class']=='STAR'][plot_columns].mean(), yerr=df_spec[df_spec['class']=='STAR'][plot_columns].std(), color=star_c, ls='--', capsize=spec_capsize, capthick=capthick, linewidth=linewidth_s, elinewidth=elinewidth, label='Stars with spectra: {0}'.format(len(df_spec[df_spec['class']=='STAR'])), transform=trans+offset(+5))
    c[-1][0].set_linestyle('--')


    plt.legend(frameon=False)
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=rotation) # custom tick labels
    plt.xlim(-0.4, len(xlabels)-0.6) # auto scale with sticky_edges=False doesn't seem to work, so do manually to give buffers at edge of plot
    plt.xlabel('Feature name')
    plt.ylabel('Magnitude')
    plt.ylim(7,26)
    #ax1.margins(x=0.4)
    #ax1.use_sticky_edges = False
    #ax1.autoscale_view(scalex=True)
    plt.tight_layout()

    # Histogram of psf_r - cmod_r. Resolved source or not?
    plt.sca(ax2)
    x, y = histvals(df_g.resolvedr.values, bins=bins_res)
    plt.plot(x, y, ls=ls, color=galaxy_c)
    x, y = histvals(df_q.resolvedr.values, bins=bins_res)
    plt.plot(x, y, ls=ls, color=quasar_c)
    x, y = histvals(df_s.resolvedr.values, bins=bins_res)
    plt.plot(x, y, ls=ls, color=star_c)
    # sources with spectra:
    x, y = histvals(df_spec[df_spec['class']=='GALAXY'].resolvedr.values, bins=bins_res)
    plt.plot(x, y, ls='--', color=galaxy_c, linewidth=0.5)
    x, y = histvals(df_spec[df_spec['class']=='QSO'].resolvedr.values, bins=bins_res)
    plt.plot(x, y, ls='--', color=quasar_c, linewidth=0.5)
    x, y = histvals(df_spec[df_spec['class']=='STAR'].resolvedr.values, bins=bins_res)
    plt.plot(x, y, ls='--', color=star_c, linewidth=0.5)

    # ticks on both sides of plot
    ax1.tick_params(which='both', right=True)
    ax2.tick_params(which='both', right=True)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('| PSF r - cmodel r | [magnitude]')
    plt.ylabel('Number')
    plt.tight_layout()
    f.savefig('New-sources-features-perclass.pdf')
    plt.clf()





    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_newsources_redshift(df_g, df_q, df_s):
    print('Plotting... New-sources-redshift.pdf')
    # bins for resolved psf_r - cmodel_r
    bins_z = 10 ** np.linspace(np.log10(1e-4), np.log10(1.1), 600)
    #bins_z = np.linspace(1e-6, 1.1, 400)
    linewidth_hist = 1

    fig = plt.figure()
    xg, yg = histvals(df_g[df_g.z>0].dropna().z, bins=bins_z)
    xq, yq = histvals(df_q[df_q.z>0].dropna().z, bins=bins_z)
    xs, ys = histvals(df_s[df_s.z>0].dropna().z, bins=bins_z)
    plt.plot(xg, yg, label='Predicted galaxies: {0}'.format(len(df_g[df_g.z>0].dropna().z.values)), ls='--', color=galaxy_c, linewidth=linewidth_hist)
    plt.plot(xq, yq, label='Predicted quasars: {0}'.format(len(df_q[df_q.z>0].dropna().z.values)), ls='--', color=quasar_c, linewidth=linewidth_hist)
    plt.plot(xs, ys, label='Predicted stars: {0}'.format(len(df_s[df_s.z>0].dropna().z.values)), ls='--', color=star_c, linewidth=linewidth_hist)

    xg, yg = histvals(df_g[df_g.photoErrorClass==1]['z'], bins=bins_z)
    xq, yq = histvals(df_q[df_q.photoErrorClass==1]['z'], bins=bins_z)
    xs, ys = histvals(df_s[df_s.photoErrorClass==1]['z'], bins=bins_z)
    plt.plot(xg, yg, label='Predicted galaxies (Error class = 1): {0}'.format(len(df_g[df_g.photoErrorClass==1].z.values)), ls=ls, color=galaxy_c, linewidth=linewidth_hist)
    plt.plot(xq, yq, label='Predicted quasars (Error class = 1): {0}'.format(len(df_q[df_q.photoErrorClass==1].z.values)), ls=ls, color=quasar_c, linewidth=linewidth_hist)
    plt.plot(xs, ys, label='Predicted stars (Error class = 1): {0}'.format(len(df_s[df_s.photoErrorClass==1].z.values)), ls=ls, color=star_c, linewidth=linewidth_hist)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Redshift')
    plt.ylabel('Number of sources')
    # ticks on both sides of plot
    plt.tick_params(axis='y', which='both', right=True)
    plt.minorticks_on()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('New-sources-redshift.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def load_spec():
    # Load in spec objects for plotting later
    print('Loading df with spec objects and appending "class_pred" and "prob_" columns...')
    df_spec = load_obj('df')
    # Load in spec df and join class_pred and prob_ columns:
    data_prep_dict_all = load_obj('data_prep_dict_all')
    classes_pred_all = load_obj('classes_pred_all')
    classes_pred_all_proba = load_obj('classes_pred_all_proba')
    # Get predicted classes from the RF classifier:
    df_predclass = pd.DataFrame(classes_pred_all, index=data_prep_dict_all['features_test'].index, columns=['class_pred'])
    # Append probabilities to the original df for test data:
    df_spec = df_spec.join(df_predclass, how='left')
    # Get probabilities from the RF classifier:
    df_proba = pd.DataFrame(classes_pred_all_proba, index=data_prep_dict_all['features_test'].index, columns=['prob_g', 'prob_q', 'prob_s'])
    # Append probabilities to the original df for test data:
    df_spec = df_spec.join(df_proba, how='left')
    return df_spec






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def print_result_numbers():
    # Count how many have probabilities above 0.9 and 0.99:
    print('Newly classified sources: ')
    print(df_all['class_pred'].value_counts())
    print(' Galaxies with prob greater than 0.9 and 0.99: {0}, {1} : as fraction of total: {2:.2f} {3:.2f}'.format( len(df_all[df_all['prob_g'] > 0.9]), len(df_all[df_all['prob_g'] > 0.99]), len(df_all[df_all['prob_g'] > 0.9])/len(df_g), len(df_all[df_all['prob_g'] > 0.99])/len(df_g) ) )
    print(' Quasars with prob greater than 0.9 and 0.99: {0} {1} : as fraction of total: {2:.2f} {3:.2f}'.format( len(df_all[df_all['prob_q'] > 0.9]), len(df_all[df_all['prob_q'] > 0.99]), len(df_all[df_all['prob_q'] > 0.9])/len(df_q), len(df_all[df_all['prob_q'] > 0.99])/len(df_q) ) )
    print(' Stars with prob greater than 0.9 and 0.99: {0} {1} : as fraction of total: {2:.2f} {3:.2f}'.format( len(df_all[df_all['prob_s'] > 0.9]), len(df_all[df_all['prob_s'] > 0.99]), len(df_all[df_all['prob_s'] > 0.9])/len(df_s), len(df_all[df_all['prob_s'] > 0.99])/len(df_s) ) )

    # Spec objects
    print('Sources classified from spectroscopic test dataset: ')
    print(df_spec['class_pred'].value_counts())
    print(' Galaxies with prob greater than 0.9 and 0.99: {0} {1} : as fraction of total: {2:.2f} {3:.2f}'.format( len(df_spec[df_spec['prob_g'] > 0.9]), len(df_spec[df_spec['prob_g'] > 0.99]), len(df_spec[df_spec['prob_g'] > 0.9])/len(df_spec[df_spec['class_pred']=='GALAXY']), len(df_spec[df_spec['prob_g'] > 0.99])/len(df_spec[df_spec['class_pred']=='GALAXY'])  ) )
    print(' Quasars with prob greater than 0.9 and 0.99: {0} {1} : as fraction of total: {2:.2f} {3:.2f}'.format( len(df_spec[df_spec['prob_q'] > 0.9]), len(df_spec[df_spec['prob_q'] > 0.99]), len(df_spec[df_spec['prob_q'] > 0.9])/len(df_spec[df_spec['class_pred']=='QSO']), len(df_spec[df_spec['prob_q'] > 0.99])/len(df_spec[df_spec['class_pred']=='QSO']) ) )
    print(' Stars with prob greater than 0.9 and 0.99: {0} {1} : as fraction of total: {2:.2f} {3:.2f}'.format( len(df_spec[df_spec['prob_s'] > 0.9]), len(df_spec[df_spec['prob_s'] > 0.99]), len(df_spec[df_spec['prob_s'] > 0.9])/len(df_spec[df_spec['class_pred']=='STAR']), len(df_spec[df_spec['prob_s'] > 0.99])/len(df_spec[df_spec['class_pred']=='STAR']) ) )






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def save_catalogue_todisk(df_all, writetodisk=True, sliced=False):
    print(' Dropping irrelevant columns...')
    # rename columns for readability
    new_colnames = ['objid', 'ra', 'dec', 'type', 'cmod_r_corr', 'psf_u_corr', 'psf_g_corr',
       'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4',
       'match_dist', 'z', 'zerr', 'photoErrorClass', 'resolvedr', 'class_pred',
       'class_prob_galaxy', 'class_prob_quasar', 'class_prob_star']
    df_all.columns = new_colnames
    print(len(df_all))

    drop_columns = ['cmod_r_corr', 'type', 'match_dist', 'z', 'zerr', 'photoErrorClass']
    df_all.drop(columns=drop_columns, inplace=True)
    # split up
    df_g = df_all[df_all['class_pred']=='GALAXY']
    df_q = df_all[df_all['class_pred']=='QSO']
    df_s = df_all[df_all['class_pred']=='STAR']

    # sort values because it's nice to have most probable ones at the top
    df_g.sort_values(by=['class_prob_galaxy'], ascending=False, inplace=True)
    df_q.sort_values(by=['class_prob_quasar'], ascending=False, inplace=True)
    df_s.sort_values(by=['class_prob_star'], ascending=False, inplace=True)

    # save as smaller dfs, aiming for less than 100 MB per df so we can upload to github

    if writetodisk==True:
        if sliced==True:
            print(' Saving galaxies...')
            number_of_chunks_g = 80
            for id, df_i in enumerate(np.array_split(df_g, number_of_chunks_g)):
               df_i.to_pickle('./data/galaxy/SDSS-ML-GALAXY-{0}.pkl'.format(id))

            print(' Saving quasars...')
            number_of_chunks_q = 4
            for id, df_i in enumerate(np.array_split(df_q, number_of_chunks_q)):
                df_i.to_pickle('./data/quasar/SDSS-ML-QSO-{0}.pkl'.format(id))

            print(' Saving stars...')
            number_of_chunks_s = 90
            for id, df_i in enumerate(np.array_split(df_s, number_of_chunks_s)):
                df_i.to_pickle('./data/star/SDSS-ML-STAR-{0}.pkl'.format(id))

        if sliced==False:
            print(' Saving data as single dfs...')
            df_all.to_pickle('./data/SDSS-ML-all.pkl')
            df_g.to_pickle('./data/SDSS-ML-galaxies.pkl')
            df_q.to_pickle('./data/SDSS-ML-quasars.pkl')
            df_s.to_pickle('./data/SDSS-ML-stars.pkl')

    return df_all






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------












if __name__ == "__main__":

    # Set plot defaults for all plots
    mpl.rcParams.update({'font.size': 8})
    mpl.rcParams.update({'figure.dpi': 400})
    # Parameters for all plots:
    quasar_c = 'hotpink'
    star_c = 'dodgerblue'
    galaxy_c = 'slategrey'
    ls = '-'
    linewidth = 1
    rotation = 0

    # Define inputs
    datafile='SDSS_allphoto_111M.csv'
    datafile_DropDuplicates='SDSS_allphoto_111M.csv_DropDuplicates'
    datafile_classified='SDSS_allphoto_111M.csv_classified'
    # feature names in the datafile
    feature_columns = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr', 'w1', 'w2', 'w3', 'w4', 'resolvedr']

    # There are 2 processing options: To batch or not to batch.
    # Preferred option is to use a high performance machine (100 GB of RAM) and avoid batch processing

    # ----------------------- ----------------------- -----------------------
    # Run in batches (for laptops, not preferred option) uncomment this section
    '''
    # Count total rows, set how many rows to load in, and how many rows per chunk
    print('counting rows in {0} ...'.format(datafile))
    with open(datafile) as f:
        total_rows = sum(1 for line in f)
    print('There are {0} rows'.format(total_rows))

    # Choose how much to read in
    #nrows = total_rows # read in everything
    nrows = 20000000 # read in a fraction (faster for testing)
    chunksize = 1000000

    # Classify new objects in batches, and save results to disk
    #new_classes_all, new_classes_proba_all = classify_new_sources_batched(datafile, rf_pipeline, feature_columns, nrows=nrows, chunksize=chunksize)
    new_classes_all = load_obj('new_classes_all')
    new_classes_proba_all = load_obj('new_classes_proba_all')

    print(len(new_classes_all[new_classes_all['class_pred']=='GALAXY']))
    print(len(new_classes_all[new_classes_all['class_pred']=='STAR']))
    print(len(new_classes_all[new_classes_all['class_pred']=='QSO']))

    plot_new_hist_batched(datafile, new_classes_all, new_classes_proba_all, feature_columns, nrows=nrows, chunksize=chunksize)
    '''
    # I haven't written batched versions of all the functions. Nor have I updated this batched version since writing final versions of the non-batched versions. Use with care.


    # ----------------------- ----------------------- -----------------------
    # High memory machine required for these functinos. Need at least ~50 GB of RAM
    # Load csv file, classify and save results as df all at once.
    print(' The following code is NOT SUITABLE FOR LAPTOPS. \n You should have a high performance machine for this with ~50+ GB of RAM to process the file with 111 million sources. \n For a laptop friendly version of this use the batched versions: e.g: "classify_new_sources_batched".')

    #drop_duplicates_function(datafile) # Takes 2 hours. then comment out. output saved to disk.

    #classify_new_sources(datafile, feature_columns) # run once, results df saved to disk
    print('Loading data file {0} ... (40+ GB of RAM is required)'.format(datafile_classified))

    # load new sources classified
    df_all = load_obj(datafile_classified)
    # Cut down data for quick test runs if needed:
    #df_all=df_all[0:100000]

    # Split out dfs per class. Doing it once here saves time when making multiple plots later
    #print('Separating classes into separate dfs...')
    #df_g = df_all[df_all['class_pred']=='GALAXY']
    #df_q = df_all[df_all['class_pred']=='QSO']
    #df_s = df_all[df_all['class_pred']=='STAR']

    # Load in spec objects for comparison and plotting
    #df_spec = load_spec()

    # Print out number of classified sources and their probabilities
    #print_result_numbers()


    # --- Plotting ---

    # Uncomment relevant plots to make:
    #plot_new_hist(df_spec, df_g, df_q, df_s)
    #plot_new_hexbin(df_g, df_q, df_s)
    #plot_new_feature_hist(df_spec, df_g, df_q, df_s)
    #plot_new_maghist(df_g, df_q, df_s)
    #plot_newsources_redshift(df_g, df_q, df_s)

    # dropping irrelevant columns and saving csv file to disk in chunks
    # turn write to disk off if you just want the cleaned up df_all
    # sliced=True will shop the df up into lots of files less than 100 MB. Not used now.
    df_all = save_catalogue_todisk(df_all, writetodisk=False, sliced=False)

    # Get small sample of sources as csv
    df_all = df_all.round(5)
    print('Sample of 100 objects saved to disk as csv: SDSS-ML-sample100.csv')
    # to get 20 of each, split up df
    sample_g = df_all[df_all.class_pred=='GALAXY'].sample(n=20)
    sample_q = df_all[df_all.class_pred=='QSO'].sample(n=20)
    sample_s = df_all[df_all.class_pred=='STAR'].sample(n=20)
    # recombine
    sample_all = pd.concat([sample_g, sample_q, sample_s])
    # save to disk as csv
    sample_all.to_csv('SDSS-ML-sample60.csv')






    # end
