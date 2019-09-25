# Written by Alex Clarke - https://github.com/informationcake/SDSS-ML
# Finds nearest neighours and makes plots.

# Pre-requisits: Run SDSS_ML.py. Obtain the following .pkl files:
# df.pkl
# data_prep_dict_all.pkl
# classes_pred_all.pkl
# classes_pred_all_proba.pkl

# Version numbers
# python: 3.6.1
# pandas: 0.25.0
# numpy: 1.17.0
# scipy: 1.3.1
# matplotlib: 3.1.1
# sklearn: 0.20.0
# datashader: 0.7.0

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
import matplotlib.image as mpimg
from matplotlib.ticker import FormatStrFormatter

import datashader as ds
from datashader.utils import export_image
from datashader import transfer_functions as tf
from datashader.colors import *

from SDSS_ML_analysis import load_obj, save_obj, prepare_classifications

# list of functions:
# get_nearest_neighbour_idx
# get_knn_accuracy_MT
# get_knn_accuracy_MT
# get_knn_accuracy
# plot_knn_f1scores



# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # multithreaded knn function used in the function "get_nearest_neighbour_idx" below
def knn_closest(vals, knn, df_vals, neighbours):
    # Find the nearest sources in feature space using previously fitted knn, and see what they were classified as with the random forest.
    closest_idx = knn.kneighbors(vals, neighbours, return_distance=False) # returns zero indexed indices.
    closest_int = closest_idx.copy() #keep integer indexes for later if we want to match to classes_pred df which is zero indexed.
    # after much pain, I've realised nn.kneighbours returns integer indices from zero in the array, and we must convert these to label indices to be matched to all other dfs. Double brackets are needed when using iloc[[]].
    for i in range(len(closest_idx)): # looping over this axis isn't needed in multiproc implementation? kept for consistency.
        for j in range(len(closest_idx[i])):
            # Must use same reference that the model was fitted to to convert indexes. This is input as a view of the original df (df_vals). Btw only using test data for knn part, ignoring training data completely.
            closest_idx[i,j] = df_vals.iloc[[closest_idx[i,j]]].index.values #when using df as ref. chaining .loc is very slow. avoid.
            # Could use data_all['features_test'] for this for a small speed up since it avoid df.loc, however this wont work when wanting to fit to the 1-D feature. So to keep it general, we use the original df for all cases.
            #closest_idx[i,j] = data_all['features_test'].iloc[[closest_idx[i,j]]].index.values

    # multithreading a function means returning multiple outputs is difficult/not possible as separate variables as it is when not using multithread. So, to generalise this, wrap up arguments into a dictionary, and unapck and sort afterwards.

    return {'closest_idx':closest_idx, 'closest_int':closest_int}






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def get_nearest_neighbour_idx(df, dim=1, sample_every_n=2000, neighbours=10000):

    # K Nearest Neighbours - find 10000 nearest points per source and return accuracy
    # Whilst KNN fitting is pretty instant, KNN finding is really slow per source for large neighbours and does not have a multithread option. Therefore, I implement a multithread version of my own. This can take a long time, so always save output to disk.

    # Choose data to fit knn to - either 1-D feaeture, or all 10 features:
    if dim==1: # reshape array in this case due to knn fitting implementation
        df_knnfit = df.loc[data_all['features_test'].index.values]['feature_1D'].values.reshape(-1,1)
    if dim==10:
        df_knnfit = df.loc[data_all['features_test'].index.values][[*feature_columns]]

    knn = NearestNeighbors(n_neighbors=neighbours, algorithm='auto', n_jobs=1) # keep n_jobs=1 else my multiprocess implementation of the knn_closest function has issues, the n_jobs tag is kept within the knn object.
    knn.fit(df_knnfit) # fit the knn to the data, this is very quick.

    # select data points to find nn to. Sample_every_n allows a much faster version of this to run.
    if dim==1:
        vals_closest = df.loc[data_all['features_test'].index.values]['feature_1D'][0::sample_every_n].values.reshape(-1,1) # reshape because it's only 1 number. [0::1000] sample every 1000 points?
    if dim==10:
        vals_closest = df.loc[data_all['features_test'].index.values][[*feature_columns]][0::sample_every_n].values

    # Pass a copy of the df used for fitting knn to the knn_closest function for referencing indices
    df_vals = df.loc[data_all['features_test'].index.values]

    num_workers = multiprocessing.cpu_count()
    print('Starting multiproc finding {0} nearest neighbours (in {1}-D) on {2} sources with {3} CPUs... could take 30mins or more if sources/CPUs is more than 100... (should linearly scale: 2X CPU = 1/2 time)'.format(neighbours, dim, len(vals_closest), num_workers))

    # because knn needs more than one entry at a time in the input array else it complains, multiprocess wont work in the standard way. So I split the master array up into segments and feed them in as an array of arrays. Perhaps there is a better way to do this but this works and should have virtually no decrease in speed. Waiting for the day sklearn knn has built in multithread into knn.neighbours().
    vals_closest_split = np.array_split(np.array(vals_closest),len(vals_closest)/20) # split into chunks containing subsets of sources

    with multiprocessing.Pool(processes=num_workers) as pool:
        closest_dictionary_list = pool.starmap(knn_closest, zip(vals_closest_split, itertools.repeat(knn), itertools.repeat(df_vals), itertools.repeat(neighbours))) #btw a starmap is a way to pass multiple arguments to a function using multi-process
        pool.close()
        pool.join()

    #closest_idx = knn_closest(vals_closest, knn, df_vals, neighbours) # Single core run? Not needed, multicore will work for all runs now.
    print('knn multiproc finished... sorting outputs...')
    # Now sort the output from multiproc
    # Initialise arrays of correct shape using first value
    closest_idx_concat = np.array( np.array(closest_dictionary_list[0]['closest_idx']) )
    closest_idx_concat_int = np.array( np.array(closest_dictionary_list[0]['closest_int']) )
    # loop over list of dictionaries, stack the _idx and _int arrrays together for all source that were initialliy split up in vals_closest_split during multiprocess.
    for i in range(1, len(closest_dictionary_list), 1): # Start from 1 because first entry already initalised above.
        closest_idx_concat = np.vstack(( closest_idx_concat, np.array(closest_dictionary_list[i]['closest_idx']) ))
        closest_idx_concat_int = np.vstack(( closest_idx_concat_int, np.array(closest_dictionary_list[i]['closest_int']) ))

    # Save output to disk
    print('Saving knn indices to disk as "closest_idx_concat_'+str(dim)+'.pkl"... ')
    save_obj(closest_idx_concat, 'closest_idx_concat_'+str(dim)+'D')
    # save_obj(closest_idx_concat_int, 'closest_idx_concat_int_'+str(dim)+'D') # Not used anymore, save time/disk space.
    # note that I don't think I need closest_idx_concat_int anymore after writing plot_knn_accuracy more effectively

    # no outputs, always save to disk






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # calculate f1 scores for each source, function is used in multiprocess get_knn_accuracy() above. Forloop version took hours because .intersection() is very slow.
def get_knn_accuracy_MT(closest):
    # test if source in question was correctly identified or not, and assign flag
    # closest_int[i,0] is the first object in the list, which is the object knn tried to find nearest neighbours for
    # test if it was correctly identified, make flag for tracking later in plots
    correct_source = 0
    if (df.loc[closest[0]]['class_pred'] == 'GALAXY') & (df.loc[closest[0]]['class'] == 'GALAXY'):
        correct_source = 1
    if (df.loc[closest[0]]['class_pred'] == 'QSO') & (df.loc[closest[0]]['class'] == 'QSO'):
        correct_source = 1
    if (df.loc[closest[0]]['class_pred'] == 'STAR') & (df.loc[closest[0]]['class'] == 'STAR'):
        correct_source = 1

    xvar='feature_1D'
    zsig=1 # fun story, I ran this for days without setting this variable.
    # ------ star ------
    tp = len(correct_star_all.index.intersection(closest))
    fp1 = len(missed_quasar_as_star_all.index.intersection(closest)) # for precision and f1
    fp2 = len(missed_galaxy_as_star_all.index.intersection(closest))
    fn1 = len(missed_star_as_quasar_all.index.intersection(closest)) # for recall and f1
    fn2 = len(missed_star_as_galaxy_all.index.intersection(closest))
    star_probs_mean = df.loc[correct_star_all.index.intersection(closest)]['prob_s'].mean()
    star_probs_std = df.loc[correct_star_all.index.intersection(closest)]['prob_s'].std()
    star_xvar_mean = df.loc[correct_star_all.index.intersection(closest)][xvar].mean()
    star_xvar_std = df.loc[correct_star_all.index.intersection(closest)][xvar].std()
    # --- F1-score ---
    try:
        f1_s = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        f1_s_err = ( zsig/( (tp+fp1+fp2+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2+fn1+fn2) / (tp+fp1+fp2+fn1+fn2)) + zsig/4 ) )
    except:
        f1_s = 0
        f1_s_err = 0

    # ------ Galaxy ------
    tp = len(correct_galaxy_all.index.intersection(closest))
    fp1 = len(missed_star_as_galaxy_all.index.intersection(closest))
    fp2 = len(missed_quasar_as_galaxy_all.index.intersection(closest))
    fn1 = len(missed_galaxy_as_star_all.index.intersection(closest))
    fn2 = len(missed_galaxy_as_quasar_all.index.intersection(closest))
    galaxy_probs_mean = df.loc[correct_galaxy_all.index.intersection(closest)]['prob_g'].mean()
    galaxy_probs_std = df.loc[correct_galaxy_all.index.intersection(closest)]['prob_g'].std()
    galaxy_xvar_mean = df.loc[correct_galaxy_all.index.intersection(closest)][xvar].mean()
    galaxy_xvar_std = df.loc[correct_galaxy_all.index.intersection(closest)][xvar].std()
    # --- F1-score ---
    try:
        f1_g = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        f1_g_err = ( zsig/( (tp+fp1+fp2+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2+fn1+fn2) / (tp+fp1+fp2+fn1+fn2)) + zsig/4 ) )
    except:
        f1_g = 0
        f1_g_err = 0

    # ------ Quasar ------
    tp = len(correct_quasar_all.index.intersection(closest))
    fp1 = len(missed_star_as_quasar_all.index.intersection(closest))
    fp2 = len(missed_galaxy_as_quasar_all.index.intersection(closest))
    fn1 = len(missed_quasar_as_star_all.index.intersection(closest))
    fn2 = len(missed_quasar_as_galaxy_all.index.intersection(closest))
    quasar_probs_mean = df.loc[correct_quasar_all.index.intersection(closest)]['prob_q'].mean()
    quasar_probs_std = df.loc[correct_quasar_all.index.intersection(closest)]['prob_q'].std()
    quasar_xvar_mean = df.loc[correct_quasar_all.index.intersection(closest)][xvar].mean()
    quasar_xvar_std = df.loc[correct_quasar_all.index.intersection(closest)][xvar].std()
    # --- F1-score ---
    try:
        f1_q = 2*tp / (2*tp + fp1+fp2 + fn1+fn2)
        f1_q_err = ( zsig/( (tp+fp1+fp2+fn1+fn2) + zsig**2) ) * ( np.sqrt( (tp*(fp1+fp2+fn1+fn2) / (tp+fp1+fp2+fn1+fn2)) + zsig/4 ) )
    except:
        f1_q = 0
        f1_q_err = 0

    return {'f1g':f1_g, 'f1q':f1_q, 'f1s':f1_s, 'f1gerr':f1_g_err, 'f1qerr':f1_q_err, 'f1serr':f1_s_err, 'galaxy_probs_mean':galaxy_probs_mean, 'galaxy_probs_std':galaxy_probs_std, 'galaxy_xvar_mean':galaxy_xvar_mean, 'galaxy_xvar_std':galaxy_xvar_std, 'quasar_probs_mean':quasar_probs_mean, 'quasar_probs_std':quasar_probs_std, 'quasar_xvar_mean':quasar_xvar_mean, 'quasar_xvar_std':quasar_xvar_std, 'star_probs_mean':star_probs_mean, 'star_probs_std':star_probs_std, 'star_xvar_mean':star_xvar_mean, 'star_xvar_std':star_xvar_std, 'correct_source':correct_source}






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






    # multiprocess function to get f1scores for each of the sources from KNN search.
def get_knn_accuracy(dim=1):
    # Load output from previous run:
    print('Loading knn indices from previous run saved on disk...')
    filename='closest_idx_concat_'+str(dim)+'D'
    closest = load_obj(filename)
    #closest_d = [closest]
    #print(closest_d)
    print('starting multiprocessing...')
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        knn_f1scores = pool.starmap(get_knn_accuracy_MT, zip(closest))
        pool.close()
        pool.join()
    save_obj(knn_f1scores, 'knn_f1scores_'+str(dim)+'D')
    return knn_f1scores






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# Using output from the function above "get_nearest_neighbour_idx", plot accuracy metrics for the nearest neighbours per source
def plot_knn_f1scores(plot_label=''):
# Plots F1-score for each source from the nearest neighbours found using knn_closest. Input is a list of indices.
# If dim==1 knn found in 1-D. If dim==10, knn found in 10-D. (see later half of this function for details)
# Choose to plot as function of 1D feature or r magnitude.
    # Load output from previous run:
    print('Loading knn indices from previous run saved on disk...')
    filename1d = 'knn_f1scores_1D'
    filename10d = 'knn_f1scores_10D'

    try:
        knn_f1scores_1d = load_obj(filename1d)
        knn_f1scores_10d = load_obj(filename10d)
    except:
        print('Failed to load knn_f1scores_*.pkl from disk - did you run "get_knn_accuracy()" yet?')
        exit()

    # combine list of dicts into single dictionary
    knn_f1scores_1d = {k:[d.get(k) for d in knn_f1scores_1d] for k in {k for d in knn_f1scores_1d for k in d}}
    knn_f1scores_10d = {k:[d.get(k) for d in knn_f1scores_10d] for k in {k for d in knn_f1scores_10d for k in d}}
    df1d = pd.DataFrame(knn_f1scores_1d)
    df10d = pd.DataFrame(knn_f1scores_10d)

    # 1D
    df1d_g = df1d[['galaxy_xvar_mean', 'galaxy_xvar_std', 'galaxy_probs_mean', 'galaxy_probs_std', 'f1g', 'f1gerr', 'correct_source']].copy()
    df1d_q = df1d[['quasar_xvar_mean', 'quasar_xvar_std', 'quasar_probs_mean', 'quasar_probs_std', 'f1q', 'f1qerr', 'correct_source']].copy()
    df1d_s = df1d[['star_xvar_mean', 'star_xvar_std', 'star_probs_mean', 'star_probs_std', 'f1s', 'f1serr', 'correct_source']].copy()
    df1d_g['class'] = 'GALAXY'
    df1d_g.columns = ['feature1d_mean', 'feature1d_std', 'probs_mean', 'probs_std', 'f1', 'f1err', 'correct_source', 'class']
    df1d_q['class'] = 'QSO'
    df1d_q.columns = ['feature1d_mean', 'feature1d_std', 'probs_mean', 'probs_std', 'f1', 'f1err', 'correct_source', 'class']
    df1d_s['class'] = 'STAR'
    df1d_s.columns = ['feature1d_mean', 'feature1d_std', 'probs_mean', 'probs_std', 'f1', 'f1err', 'correct_source', 'class']
    df_all_1d = pd.concat([df1d_g, df1d_q, df1d_s], axis=0)
    df_all_1d['class'] = df_all_1d['class'].astype('category') # datashader wants categorical class

    df10d_g = df10d[['galaxy_xvar_mean', 'galaxy_xvar_std', 'galaxy_probs_mean', 'galaxy_probs_std', 'f1g', 'f1gerr', 'correct_source']].copy()
    df10d_q = df10d[['quasar_xvar_mean', 'quasar_xvar_std', 'quasar_probs_mean', 'quasar_probs_std', 'f1q', 'f1qerr', 'correct_source']].copy()
    df10d_s = df10d[['star_xvar_mean', 'star_xvar_std', 'star_probs_mean', 'star_probs_std', 'f1s', 'f1serr', 'correct_source']].copy()
    df10d_g['class'] = 'GALAXY'
    df10d_g.columns = ['feature10d_mean', 'feature10d_std', 'probs_mean', 'probs_std', 'f1', 'f1err', 'correct_source','class']
    df10d_q['class'] = 'QSO'
    df10d_q.columns = ['feature10d_mean', 'feature10d_std', 'probs_mean', 'probs_std', 'f1', 'f1err', 'correct_source','class']
    df10d_s['class'] = 'STAR'
    df10d_s.columns = ['feature10d_mean', 'feature10d_std', 'probs_mean', 'probs_std', 'f1', 'f1err', 'correct_source','class']
    df_all_10d = pd.concat([df10d_g, df10d_q, df10d_s], axis=0)
    df_all_10d['class'] = df_all_10d['class'].astype('category') # datashader wants categorical class

    # Did we fit the knn in 1-D or in 10-D?
    # In 1-D a few thousand nearest neighbours will likely be a healthy mix of the 3 classes throughout most/all of the feature space. So you will get reliable numbers for F1 scores per class (perhaps with differring error bars). These are basically a round-about way of getting F1 scores shown in the histogram created by the function plot_histogram_matrix_f1. It is nice they agree (they most definately should). The mannor in which they agree is interesting - since knn effectively uses variable bin widths to get enough nearest neighbours, whilst plot_histogram_matrix_f1 uses fixed bin widths and averages within that bin.

    # select correct sources only?
    # Only plot f1-score for correct object type in question. e.g. If it's a galaxy, nearest 10000 objects will likely only be galaxies, so f1 for star and quasar will be very poor or zero because there are no True Positives in this area of 1-D feature space. In 1-D feature space the 10000 nearest neighbours were a healthy mix of all three classes so we didn't have this problem.

    print(df_all_1d.correct_source.value_counts())
    print(df_all_10d.correct_source.value_counts())
    df_all_1d = df_all_1d[df_all_1d.correct_source==1]
    df_all_10d = df_all_10d[df_all_10d.correct_source==1]

    # only 5000 sources are wrong, not so bad.
    # Create datashader pngs for each plot, since we have too much data for matplotlib to handle

    # 1D - 1dfeature vs f1
    xmin1d = df1d.star_xvar_mean.min() - 0.1 # padd for plotting later
    xmax1d = df1d.star_xvar_mean.max() + 0.1
    ymin = 0
    ymax = 1.05
    cvs = ds.Canvas(plot_width=1000, plot_height=600,
                   x_range=(xmin1d,xmax1d), y_range=(ymin,ymax),
                   x_axis_type='linear', y_axis_type='linear')
    agg = cvs.points(df_all_1d, 'feature1d_mean', 'f1', ds.count_cat('class'))
    ckey = dict(GALAXY='slategrey', QSO='hotpink', STAR='dodgerblue')
    img = tf.shade(agg, color_key=ckey, how='log')
    export_image(img, 'knn1d_1d_vs_f1', fmt='.png', background='white')

    # 10D - 1dfeature vs f1
    xmin10d = df10d.star_xvar_mean.min() - 0.1 # padd for plotting later
    xmax10d = df10d.star_xvar_mean.max() + 0.1
    ymin = 0
    ymax = 1.05
    cvs = ds.Canvas(plot_width=200, plot_height=120,
                   x_range=(xmin10d,xmax10d), y_range=(ymin,ymax),
                   x_axis_type='linear', y_axis_type='linear')
    agg = cvs.points(df_all_10d, 'feature10d_mean', 'f1', ds.count_cat('class'))
    ckey = dict(GALAXY='slategrey', QSO='hotpink', STAR='dodgerblue')
    img = tf.shade(agg, color_key=ckey, how='log')
    export_image(img, 'knn10d_1d_vs_f1', fmt='.png', background='white')

    # 1D - prob vs f1
    xmin1d_probs = 0 # padd for plotting later
    xmax1d_probs = 1.05
    ymin = 0
    ymax = 1.05
    cvs = ds.Canvas(plot_width=300, plot_height=300,
                   x_range=(xmin1d_probs,xmax1d_probs), y_range=(ymin,ymax),
                   x_axis_type='linear', y_axis_type='linear')
    agg = cvs.points(df_all_1d, 'probs_mean', 'f1', ds.count_cat('class'))
    ckey = dict(GALAXY='slategrey', QSO='hotpink', STAR='dodgerblue')
    img = tf.shade(agg, color_key=ckey, how='log')
    export_image(img, 'knn1d_probs_vs_f1', fmt='.png', background='white')

    # 10D - 1dfeature vs f1
    xmin10d_probs = 0 # padd for plotting later
    xmax10d_probs = 1.05
    ymin = 0
    ymax = 1.05
    cvs = ds.Canvas(plot_width=200, plot_height=200,
                   x_range=(xmin10d_probs,xmax10d_probs), y_range=(ymin,ymax),
                   x_axis_type='linear', y_axis_type='linear')
    agg = cvs.points(df_all_10d, 'probs_mean', 'f1', ds.count_cat('class'))
    ckey = dict(GALAXY='slategrey', QSO='hotpink', STAR='dodgerblue')
    img = tf.shade(agg, color_key=ckey, how='log')
    export_image(img, 'knn10d_probs_vs_f1', fmt='.png', background='white')





    # ----------------- plotting -----------------
    # get datashader pngs, and plot a small sample of points over the top to guide eye with error bars.
    img_1d_1d = mpimg.imread('knn1d_1d_vs_f1.png')
    img_1d_probs = mpimg.imread('knn1d_probs_vs_f1.png')
    mpl.rcParams.update({'font.size': 10})
    markeredgewidth=0.5
    mew=0.5
    elinewidth=0.5

    fig, axs = plt.subplots(1, 2, figsize=(14.5,4))
    # --- 1D --- 1d ---
    plt.sca(axs[0])
    plt.imshow(img_1d_1d, extent=[xmin1d,xmax1d,ymin*10,ymax*10]) # make yaxis 10 times larger
    # fix ylabels after scaling the axis
    ylabels = axs[0].get_yticks()
    new_ylabels = [l/10 for l in ylabels] # account for factor of 10 increase
    axs[0].set_yticklabels(new_ylabels)
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plot sample over the top to get a feel for error bars
    samp=2500
    plt.errorbar(df1d_g[0::samp]['feature1d_mean'], df1d_g[0::samp]['f1']*10, xerr=df1d_g[0::samp]['feature1d_std'], yerr=df1d_g[0::samp]['f1err']*10, color=galaxy_c, elinewidth=elinewidth, markeredgewidth=mew, ls='none', label='Galaxies')
    plt.errorbar(df1d_q[0::samp]['feature1d_mean'], df1d_q[0::samp]['f1']*10, xerr=df1d_q[0::samp]['feature1d_std'], yerr=df1d_q[0::samp]['f1err']*10, color=quasar_c, elinewidth=elinewidth, markeredgewidth=mew, ls='none', label='Quasars')
    plt.errorbar(df1d_s[0::samp]['feature1d_mean'], df1d_s[0::samp]['f1']*10, xerr=df1d_s[0::samp]['feature1d_std'], yerr=df1d_s[0::samp]['f1err']*10, color=star_c, elinewidth=elinewidth, markeredgewidth=mew, ls='none', label='Stars')

    plt.tick_params(axis='y', which='both', right=True)
    plt.minorticks_on()
    plt.xlabel('1D feature')
    plt.ylabel('F1 score in 1 dimensions')
    #axs[1].text(0.95, 0.01, 'calculated from 10000 nearest neighbours in 10 dimensions', verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes, color='black', fontsize=8)
    plt.xlim(-8,11.5)
    plt.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    fig.tight_layout()

    # --- 1D --- probs ---
    plt.sca(axs[1])
    xf=2
    plt.imshow(img_1d_probs, extent=[xmin1d_probs*xf,xmax1d_probs*xf,ymin,ymax]) # make xaxis larger
    # fix ylabels after scaling the axis
    #xlabels = axs[0].get_xticks()
    #new_xlabels = [l/xf for l in xlabels] # account for scaling axis
    axs[1].set_xticks(np.arange(0,2.1,step=0.2))
    axs[1].set_xticklabels(np.arange(0,1.1,step=0.1))
    #axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # doesn't work
    # getting some labels with 8 F****** decimal places without these two lines:
    labels = [item.get_text() for item in axs[1].get_xticklabels()]
    axs[1].set_xticklabels([str(round(float(label), 2)) for label in labels])


    # plot sample over the top to get a feel for error bars
    df1d_g2=df1d_g[(df1d_g.f1 < 0.85) & (df1d_g.probs_mean < 0.85)][0::3000]
    plt.errorbar(df1d_g2['probs_mean']*xf, df1d_g2['f1'], xerr=df1d_g2['probs_std']*xf, yerr=df1d_g2['f1err'], color=galaxy_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Galaxies')
    df1d_q2=df1d_q[(df1d_q.f1 < 0.85) & (df1d_q.probs_mean < 0.85)][0::3000]
    plt.errorbar(df1d_q2['probs_mean']*xf, df1d_q2['f1'], xerr=df1d_q2['probs_std']*xf, yerr=df1d_q2['f1err'], color=quasar_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Quasars')
    df1d_q2=df1d_q[(df1d_q.f1 < 0.85) & (df1d_q.probs_mean < 0.75)][0::800] # plot more at lower values in undersampled region
    plt.errorbar(df1d_q2['probs_mean']*xf, df1d_q2['f1'], xerr=df1d_q2['probs_std']*xf, yerr=df1d_q2['f1err'], color=quasar_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew)
    df1d_s2=df1d_s[(df1d_s.f1 < 0.85) & (df1d_s.probs_mean < 0.85)][0::3000]
    plt.errorbar(df1d_s2['probs_mean']*xf, df1d_s2['f1'], xerr=df1d_s2['probs_std']*xf, yerr=df1d_s2['f1err'], color=star_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Stars')

    plt.tick_params(axis='y', which='both', right=True)
    plt.minorticks_on()
    plt.xlabel('Classification probability')
    plt.ylabel('F1 score in 1 dimension')
    #axs[0].text(0.95, 0.01, 'calculated from 10000 nearest neighbours in 1 dimension', verticalalignment='bottom', horizontalalignment='right', transform=axs[0].transAxes, color='black', fontsize=8)
    plt.tight_layout()

    #fig.subplots_adjust(wspace=0.1, hspace=0.1) # Must come after tight_layout to work! ... doesn't seem to work when using imshow :(
    fig.savefig('knn_plot_1dfeature'+plot_label+'.pdf')
    plt.clf()






    # ---------------- Now plot probabilities ----------------
    # ---------------- Now plot probabilities ----------------
    # ---------------- Now plot probabilities ----------------






    # ----------------- plotting -----------------
    elinewidth=0.2
    mpl.rcParams.update({'font.size': 10}) # else its really small in the paper

    img_10d_1d = mpimg.imread('knn10d_1d_vs_f1.png')
    img_10d_probs = mpimg.imread('knn10d_probs_vs_f1.png')

    fig, axs = plt.subplots(1, 2, figsize=(14.5,4))
    xf=2 # make x-axis twice as long as y.

    # --- 10D ---
    plt.sca(axs[0])
    plt.imshow(img_10d_1d, extent=[xmin10d,xmax10d,ymin*10,ymax*10]) # make yaxis 10 times larger
    # fix ylabels after scaling the axis
    ylabels = axs[0].get_yticks()
    new_ylabels = [l/10 for l in ylabels] # account for factor of 10 increase
    axs[0].set_yticklabels(new_ylabels)
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plot sample over the top to get a feel for error bars
    df10d_g2=df10d_g[df10d_g.f1 < 0.95][0::500] # only plot error bars below 0.95 because above this they are v small.
    plt.errorbar(df10d_g2['feature10d_mean'], df10d_g2['f1']*10, xerr=df10d_g2['feature10d_std'], yerr=df10d_g2['f1err']*10, color=galaxy_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Galaxies')
    df10d_q2=df10d_q[df10d_q.f1 < 0.95][0::500]
    plt.errorbar(df10d_q2['feature10d_mean'], df10d_q2['f1']*10, xerr=df10d_q2['feature10d_std'], yerr=df10d_q2['f1err']*10, color=quasar_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Quasars')
    df10d_s2=df10d_s[df10d_s.f1 < 0.95][0::500]
    plt.errorbar(df10d_s2['feature10d_mean'], df10d_s2['f1']*10, xerr=df10d_s2['feature10d_std'], yerr=df10d_s2['f1err']*10, color=star_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Stars')
    plt.tick_params(axis='y', which='both', right=True)
    plt.minorticks_on()
    plt.xlabel('1D feature')
    plt.ylabel('F1 score in 10 dimensions')
    #axs[1].text(0.95, 0.01, 'calculated from 10000 nearest neighbours in 10 dimensions', verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes, color='black', fontsize=8)
    plt.xlim(-8,11.5)
    plt.tight_layout()

    # --- 10D --- probs ---
    plt.sca(axs[1])
    plt.imshow(img_10d_probs, extent=[xmin10d_probs*xf,xmax10d_probs*xf,ymin,ymax]) # make xaxis larger
    # fix ylabels after scaling the axis
    #xlabels = axs[1].get_xticks()
    #new_xlabels = [l/xf for l in xlabels] # account for scaling axis
    #axs[1].set_xticklabels(new_xlabels)
    axs[1].set_xticks(np.arange(0,2.1,step=0.2))
    axs[1].set_xticklabels(np.arange(0,1.1,step=0.1))
    #axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # doesn't work
    labels = [item.get_text() for item in axs[1].get_xticklabels()]
    axs[1].set_xticklabels([str(round(float(label), 2)) for label in labels])

    # plot sample over the top to get a feel for error bars
    df10d_g2=df10d_g[(df10d_g.f1 < 0.85) & (df10d_g.probs_mean < 0.85)][0::1000] # only plot error bars below 0.95 because above this they are v small, and overcrowd the plot.
    plt.errorbar(df10d_g2['probs_mean']*xf, df10d_g2['f1'], xerr=df10d_g2['probs_std']*xf, yerr=df10d_g2['f1err'], color=galaxy_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Galaxy')
    df10d_q2=df10d_q[(df10d_q.f1 < 0.85) & (df10d_q.probs_mean < 0.85)][0::1000]
    plt.errorbar(df10d_q2['probs_mean']*xf, df10d_q2['f1'], xerr=df10d_q2['probs_std']*xf, yerr=df10d_q2['f1err'], color=quasar_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Quasar')
    df10d_s2=df10d_s[(df10d_s.f1 < 0.85) & (df10d_s.probs_mean < 0.85)][0::1000]
    plt.errorbar(df10d_s2['probs_mean']*xf, df10d_s2['f1'], xerr=df10d_s2['probs_std']*xf, yerr=df10d_s2['f1err'], color=star_c, elinewidth=elinewidth, ls='none', markeredgewidth=mew, label='Star')

    plt.tick_params(axis='y', which='both', right=True)
    plt.minorticks_on()
    plt.xlabel('Classification probability')
    plt.ylabel('F1 score in 10 dimensions')
    plt.legend(frameon=False, loc='upper left')
    #axs[1].text(0.95, 0.01, 'calculated from 10000 nearest neighbours in 10 dimensions', verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes, color='black', fontsize=8)
    plt.tight_layout()
    fig.tight_layout()
    fig.savefig('knn_plot_probs'+plot_label+'.pdf')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#-------------------------------------------------
# Main
#-------------------------------------------------






if __name__ == "__main__": #so you can import this code and run by hand if desired

    # define some variables and prepare some data, much like in SDSS_ML_analysis.py

    # Set plot defaults for all plots
    mpl.rcParams.update({'font.size': 8})
    mpl.rcParams.update({'figure.dpi': 400})

    # Parameters for all plots:
    quasar_c = 'hotpink'
    star_c = 'dodgerblue'
    galaxy_c = 'slategrey'

    quasar_c2 = 'black'
    star_c2 = 'blue'
    galaxy_c2 = 'orange'

    psf_ext = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr']
    wise = ['w1' ,'w2', 'w3', 'w4']
    feature_columns = psf_ext + wise + ['resolvedr']

    df = load_obj('df')
    data_prep_dict_all = load_obj('data_prep_dict_all')
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

    # split correct and missed sources as separate arrays
    data_all, missed_star_all, missed_quasar_all, missed_galaxy_all, correct_star_all, correct_quasar_all, correct_galaxy_all, missed_star_as_quasar_all, missed_star_as_galaxy_all, missed_quasar_as_star_all, missed_quasar_as_galaxy_all, missed_galaxy_as_star_all, missed_galaxy_as_quasar_all = prepare_classifications(data_prep_dict_all, classes_pred_all, ttsplit=True)



    # ------  ------ Run k nearest neighbours in 1-D or 10-D and return F1-score  ------ ------
    # BEWARE! This is very computationally intensive. Get as many cores for multithread as possible



    # Get indices for nearest neighbours. Run once and save output to disk for plot_knn_accuracy
    # This can take a week to run for 100000 sources (when n=12). Took 22 hours (using 40 threads) in 1 and 10-D for 100000 sources.
    print('Getting NN in 1D... {0}'.format(datetime.datetime.utcnow()))
    get_nearest_neighbour_idx(df, dim=1, sample_every_n=12, neighbours=10000) # sample_every_n=2000 gives ~600 sources
    print('Getting NN in 10D... {0}'.format(datetime.datetime.utcnow()))
    get_nearest_neighbour_idx(df, dim=10, sample_every_n=12, neighbours=10000) # sample_every_n=200 gives ~6000 sources
    print('Done. {0}'.format(datetime.datetime.utcnow()))

    # Now calculate scores from those nearest neighbours
    print('Getting NN f1scores, 1D... {0}'.format(datetime.datetime.utcnow()))
    get_knn_accuracy(dim=1) # takes 20 mins
    print('--'*20)
    print('Getting NN f1scores, 10D... {0}'.format(datetime.datetime.utcnow()))
    get_knn_accuracy(dim=10) # takes 20 mins
    print('Done. {0}'.format(datetime.datetime.utcnow()))

    # Finally, make plots of those scores vs 1D feature and probabilities
    plot_knn_f1scores()






    # end
