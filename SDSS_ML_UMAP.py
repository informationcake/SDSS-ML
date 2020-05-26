# Written by Alex Clarke - https://github.com/informationcake/SDSS-ML
# It will perform dimension reduction with UMAP, and generate plots.

# Pre-requisits: Run SDSS_ML.py to obtain the clean df.

# Version numbers
# python: 3.6.1
# pandas: 0.25.0
# numpy: 1.17.0
# scipy: 1.3.1
# matplotlib: 3.1.1
# sklearn: 0.20.0
# datashader: 0.7.0

# Import functions
import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg',warn=False, force=True) #set MPL backend.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle #save/load python data objects (dictionaries/arrays)
import multiprocessing
import itertools
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import datetime
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

# Previous functions from SDSS_ML_analysis.py
from SDSS_ML_analysis import load_obj, save_obj, histvals, make_cmap
from SDSS_ML import metrics, prepare_data

#from sklearn.manifold import TSNE #single core TSNE, sklearn.
#from MulticoreTSNE import MulticoreTSNE as multiTSNE #multicore TSNE, not sklearn implementation. Must be installed separately. # not used here since UMAP is far better.
import umap
import datashader as ds
import datashader.transfer_functions as tf
from datashader.transfer_functions import shade, stack
from functools import partial
from datashader.utils import export_image
from datashader.colors import *

# list of functions:

# run_umap_spec
# run_umap_photo
# make_cmap
# reverse_colourmap

# plot_umap_ds_SpecObjs_classes
# plot_umap_ds_SpecObjs_probs
# plot_umap_ds_SpecObjs_resolvedr
# plot_umap_ds_SpecObjs_uz
# plot_umap_ds_SpecObjs_w1w2

# plot_umap_ds_PhotoObjs_classes
# plot_umap_ds_PhotoObjs_probs
# plot_umap_ds_PhotoObjs_resolvedr
# plot_umap_ds_PhotoObjs_uz
# plot_umap_ds_PhotoObjs_w1w2



# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def run_umap_spec(df, feature_columns, label='unknown', n_neighbors=15, supervised=True, sample_train=True):
    # training fraction is hardcoded to whatever was done with the random forest (from predicted data column)
    # get rows which were used in the RF training - will not have entries in class_pred
    df_train = df.loc[df['class_pred'].isnull()]
    df_test = df.loc[df['class_pred'].notnull()]

    # training on too much data can cause umap to loose global structures
    if sample_train==True:
        print('Selecting half the training data...')
        # downsample galaxies, there too many, it confuse UMAP
        #df_train_g = df_train[df_train['class']=='GALAXY'].sample(frac=0.5)
        #df_train_q = df_train[df_train['class']=='QSO']
        #df_train_s = df_train[df_train['class']=='STAR']
        #df_train = df_train[0::2] # half as much data
        df_train = df_train.sample(frac=0.5)
        #df_train = pd.concat([df_train_g, df_train_q, df_train_s])

    print('Clustering with UMAP')
    if supervised==False:
        print('Doing unsupervised UMAP')
        print('Fitting to {0} data points...'.format(len(df_train)))
        u_model = umap.UMAP(random_state=42, n_neighbors=n_neighbors).fit(df_train[feature_columns])
        save_obj(u_model, 'umap_model_unsup'+label) # save for use on photometric objects
    if supervised==True:
        print('Doing supervised UMAP')
        print('Fitting to {0} data points...'.format(len(df_train)))
        u_model = umap.UMAP(random_state=42, n_neighbors=n_neighbors).fit(df_train[feature_columns], y=df_train['class_i'])
        save_obj(u_model, 'umap_model_sup'+label) # save for use on photometric objects

    u_train = u_model.transform(df_train[feature_columns])
    u_test = u_model.transform(df_test[feature_columns])

    #u = pd.DataFrame(u, columns=['x', 'y'], index=data_prep_dict_all['features_train'].index) # index must match original df, particularly if sub-sampled
    u_train_df = pd.DataFrame(u_train, columns=['x', 'y'], index=df_train.index) # index must match original df, particularly if sub-sampled
    u_test_df = pd.DataFrame(u_test, columns=['x', 'y'], index=df_test.index) # index must match original df, particularly if sub-sampled
    u = pd.concat([u_train_df, u_test_df]) # joins the two dfs together
    df = df.join(u, how='left') # join UMAP projection to original df
    df['class_cat'] = df['class'].astype('category') # datashader requires catagorical type for colour labels.

    return df






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def run_umap_photo(df, feature_columns, label='unknown', n_neighbors=15, labels=False):

    print('Clustering with UMAP')
    print('Fitting to {0} data points...'.format(len(df)))
    if labels==False:
        print('Doing unsup umap...')
        u_model = umap.UMAP(random_state=42, n_neighbors=n_neighbors).fit(df[feature_columns])
    if labels==True:
        print('Doing sup umap...')
        u_model = umap.UMAP(random_state=42, n_neighbors=n_neighbors).fit(df[feature_columns], y=df['class_i'])
    print('Transforming sources from fitted model...')
    u_train = u_model.transform(df[feature_columns])
    u_train_df = pd.DataFrame(u_train, columns=['x_', 'y_'], index=df.index) # index must match original df, particularly if sub-sampled
    df = df.join(u_train_df, how='left') # join UMAP projection to original df
    df['class_cat'] = df['class_pred'].astype('category') # datashader requires catagorical type for colour labels.

    return df






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






def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

    #greenyellow springgreen






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_SpecObjs_classes(df, sup, label='unknown'):
    # Plotting: use datashader
    df_train = df.loc[df['class_pred'].isnull()] # get rows which were used in the RF training - will not have entries in class_pred\
    #df_train = df_train[df_train.psf_r>0]
    df_test = df.loc[df['class_pred'].notnull()]
    #df_test = df_test[df_test.psf_r>0]
    # Plot main figure. Save images for both train and test sets
    for dfs, label2 in zip([df_train, df_test], ['train', 'test']):

        # create png
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(dfs, 'x', 'y', ds.count_cat('class_cat'))
        ckey = dict(GALAXY=(101,236,101), QSO='hotpink', STAR='dodgerblue')
        #cm = partial(colormap_select, reverse=('black'!="black"))
        img = tf.shade(agg, color_key=ckey, how='log')
        export_image(img, 'UMAP-'+label+'-'+label2+'-RFclasslabels', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,10)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-'+label2+'-RFclasslabels.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        '''
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.Greys), orientation='horizontal', label='Number of sources', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        '''
        # legend for class labels
        g_leg = plt.Line2D((0,0),(0,0), color='lightgreen', marker='', linestyle='', label='Galaxies')
        q_leg = plt.Line2D((0,0),(0,0), color='hotpink', marker='', linestyle='', label='Quasars')
        s_leg = plt.Line2D((0,0),(0,0), color='dodgerblue', marker='', linestyle='', label='Stars')
        leg = plt.legend([g_leg, q_leg, s_leg], ['Galaxies', 'Quasars', 'Stars'], frameon=False)
        leg_texts = leg.get_texts()
        leg_texts[0].set_color('lightgreen')
        leg_texts[1].set_color('hotpink')
        leg_texts[2].set_color('dodgerblue')

        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-'+label2+'-RFclasslabels.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_SpecObjs_probs(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader
    #df_train = df.loc[df['class_pred'].isnull()] # get rows which were used in the RF training - will not have entries in class_pred\
    df_test = df.loc[df['class_pred'].notnull()]
    #df_test = df_test[df_test.psf_r>0]

    # Plot probability mean
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df_test, 'x', 'y', ds.mean('prob_best'))
    img = tf.shade(agg, cmap=prob_mean_c, how='log')
    export_image(img, 'UMAP-'+label+'-probs-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-probs-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    vmin = agg.data[np.isfinite(agg.data)].min() # isfinite to ignore nans
    vmax = agg.data[np.isfinite(agg.data)].max()
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted sources (test dataset)', pad=0.01, cax=cax)
    formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
    cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
    cax.xaxis.set_minor_formatter(formatter)
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-probs-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot probability STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df_test, 'x', 'y', ds.std('prob_best'))
    img = tf.shade(agg, cmap=prob_std_c, how='log')
    export_image(img, 'UMAP-'+label+'-probs-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-probs-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a.data)].min() # drop nans
    vmax = a[np.isfinite(a.data)].max()
    #vmin=9e-4 # min value is 0.001 so set a little smaller for cbar ticks to be clearer. can't be zero for cbar scale.
    print(vmin, vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    #cax.tick_params(which='major', direction='out', length=4)

    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-probs-std.pdf', bbox_inches='tight')
    plt.close(fig)

    '''
    # Plot mean stack image
    # Can i edit this colour select out?
    #colors_g = [(1,1,1), (112/255,128/255,144/255)]
    #cmap_g = make_cmap(colors_g)
    colors_g = [(1,1,1), (144/255,238/255,144/255)]
    cmap_g = make_cmap(colors_g)
    colors_q = [(1,1,1), (255/255,105/255,180/255)]
    cmap_q = make_cmap(colors_q)
    colors_s = [(1,1,1), (30/255,144/255,255/255)]
    cmap_s = make_cmap(colors_s)

    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df_test, 'x', 'y', ds.mean('prob_best'))
    agg_g = cvs.points(df_test, 'x', 'y', ds.mean('prob_g'))
    agg_q = cvs.points(df_test, 'x', 'y', ds.mean('prob_q'))
    agg_s = cvs.points(df_test, 'x', 'y', ds.mean('prob_s'))
    img = stack( tf.shade(agg_g, cmap=cmap_g, how='log', alpha=100),
                 tf.shade(agg_q, cmap=cmap_q, how='log', alpha=100),
                 tf.shade(agg_s, cmap=cmap_s, how='log', alpha=100))
    export_image(img, 'UMAP'+label+'-meanstack', fmt='.png', background='black')
    '''

    if GQSsplit==True:
        # ------ GALAXIES ------
        # Plot mean prob
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[ (df_test.class_pred=='GALAXY') ], 'x', 'y', ds.mean('prob_g'))
        img = tf.shade(agg, cmap=prob_mean_c, how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-probs-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-probs-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
        cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
        cax.xaxis.set_minor_formatter(formatter)
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-probs-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)


        # Plot galaxy std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[ (df_test.class_pred=='GALAXY') ], 'x', 'y', ds.std('prob_g'))
        img = tf.shade(agg, cmap=prob_std_c, how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-probs-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-probs-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-probs-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)


        # ------ QUASARS ------
        # Plot quasar mean probs
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='QSO'], 'x', 'y', ds.mean('prob_q'))
        img = tf.shade(agg, cmap=prob_mean_c, how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-probs-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-probs-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted quasars (test dataset)', pad=0.01, cax=cax)
        formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
        cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
        cax.xaxis.set_minor_formatter(formatter)
        #cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-probs-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)


        # Plot quasar std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='QSO'], 'x', 'y', ds.std('prob_q'))
        img = tf.shade(agg, cmap=prob_std_c, how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-probs-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-probs-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-probs-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)


        # ------ STARS ------
        # Plot star probs
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='STAR'], 'x', 'y', ds.mean('prob_s'))
        img = tf.shade(agg, cmap=prob_mean_c, how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-probs-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-probs-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted stars (test dataset)', pad=0.01, cax=cax)
        formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
        cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
        cax.xaxis.set_minor_formatter(formatter)
        #cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-probs-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot star std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='STAR'], 'x', 'y', ds.std('prob_s'))
        img = tf.shade(agg, cmap=prob_std_c, how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-probs-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-probs-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 1.01e-3
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-probs-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_SpecObjs_resolvedr(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader
    df_train = df.loc[df['class_pred'].isnull()] # get rows which were used in the RF training - will not have entries in class_pred\
    df_test = df.loc[df['class_pred'].notnull()]
    #df_test = df_test[df_test.psf_r>0]
    # ------ ALL SOURCES ------

    # Plot resolvedr MEAN
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df_test, 'x', 'y', ds.mean('resolvedr'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-'+label+'-resolvedr-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-resolvedr-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    vmin=7e-6 # keep cbar neat in boundary otherwise labels overlap
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-resolvedr-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot resolvedr STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df_test, 'x', 'y', ds.std('resolvedr'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-'+label+'-resolvedr-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-resolvedr-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-resolvedr-std.pdf', bbox_inches='tight')
    plt.close(fig)

    if GQSsplit==True:
        # ------ GALAXIES ------
        # Plot resolvedr MEAN for galaxies
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[df_test.class_pred=='GALAXY'], 'x', 'y', ds.mean('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-resolvedr-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-resolvedr-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-resolvedr-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot resolvedr STD for galaxies
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[df_test.class_pred=='GALAXY'], 'x', 'y', ds.std('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-resolvedr-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-resolvedr-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-resolvedr-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # ------ QUASARS ------
        # Plot resolvedr MEAN for quasars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[df_test.class_pred=='QSO'], 'x', 'y', ds.mean('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-resolvedr-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-resolvedr-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 7e-6
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-resolvedr-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot resolvedr STD for quasars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[df_test.class_pred=='QSO'], 'x', 'y', ds.std('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-resolvedr-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-resolvedr-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-resolvedr-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)


        # ------ STARS ------
        # Plot resolvedr MEAN for stars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[df_test.class_pred=='STAR'], 'x', 'y', ds.mean('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-resolvedr-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-resolvedr-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 7e-6
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-resolvedr-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot resolvedr STD for stars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df_test[df_test.class_pred=='STAR'], 'x', 'y', ds.std('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-resolvedr-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-resolvedr-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-resolvedr-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_SpecObjs_uz(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader
    #df_train = df.loc[df['class_pred'].isnull()] # get rows which were used in the RF training - will not have entries in class_pred\
    df_test = df.loc[df['class_pred'].notnull()]
    #df_test = df_test[df_test.psf_r>0]

    # ------ ALL SOURCES ------

    # Plot uz mean
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
    agg = cvs.points(df_test, 'x', 'y', ds.mean('SDSS_si'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-'+label+'-uz-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-uz-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-uz-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot uz STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
    agg = cvs.points(df_test, 'x', 'y', ds.std('SDSS_si'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-'+label+'-uz-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-uz-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    vmin=7e-6 # keep fig neat
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| magnitude for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-uz-std.pdf', bbox_inches='tight')
    plt.close(fig)


    if GQSsplit==True:
        # ----- GALAXIES -----

        # Plot uz MEAN
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'GALAXY'], 'x', 'y', ds.mean('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-uz-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-uz-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-uz-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz STD
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'GALAXY'], 'x', 'y', ds.std('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-uz-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-uz-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-uz-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- QUASARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'QSO'], 'x', 'y', ds.mean('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-uz-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-uz-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-uz-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'QSO'], 'x', 'y', ds.std('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-uz-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-uz-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-uz-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- STARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'STAR'], 'x', 'y', ds.mean('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-uz-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-uz-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmax = 13
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-uz-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_si'] = np.sqrt( (df_test['psf_u'] - df_test['psf_z'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'STAR'], 'x', 'y', ds.std('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-uz-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-uz-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        if sup=='sup':
            vmin=1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-uz-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_SpecObjs_w1w2(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader
    #df_train = df.loc[df['class_pred'].isnull()] # get rows which were used in the RF training - will not have entries in class_pred\
    df_test = df.loc[df['class_pred'].notnull()]
    #df_test = df_test[df_test.psf_r>0]

    # ------ ALL SOURCES ------

    # Plot uz mean
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
    agg = cvs.points(df_test, 'x', 'y', ds.mean('SDSS_wi'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-'+label+'-w1w2-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-w1w2-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-w1w2-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot uz STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
    agg = cvs.points(df_test, 'x', 'y', ds.std('SDSS_wi'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-'+label+'-w1w2-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'-w1w2-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    vmax=2
    #vmin=7e-6 # keep fig neat
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| magnitude for predicted sources (test dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'-w1w2-std.pdf', bbox_inches='tight')
    plt.close(fig)


    if GQSsplit==True:
        # ----- GALAXIES -----

        # Plot uz MEAN
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'GALAXY'], 'x', 'y', ds.mean('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-w1w2-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-w1w2-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 8e-4
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-w1w2-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz STD
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'GALAXY'], 'x', 'y', ds.std('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-w1w2-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-w1w2-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmax = 2
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| for predicted galaxies (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-w1w2-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- QUASARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'QSO'], 'x', 'y', ds.mean('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-w1w2-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-w1w2-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=8e-4
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-w1w2-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'QSO'], 'x', 'y', ds.std('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-w1w2-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-w1w2-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=8e-4
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| for predicted quasars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-w1w2-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- STARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'STAR'], 'x', 'y', ds.mean('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-w1w2-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-w1w2-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=1.01e-3
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-w1w2-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df_test['SDSS_wi'] = np.sqrt( (df_test['w1'] - df_test['w2'])**2 )
        agg = cvs.points(df_test[df_test.class_pred == 'STAR'], 'x', 'y', ds.std('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-'+label+'-w1w2-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-'+label+'-w1w2-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmax = 2
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| for predicted stars (test dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-'+label+'-w1w2-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)


















# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
# photo plot functions



















def plot_umap_ds_PhotoObjs_classes(df, sup, label='unknown'):
    # Plotting: use datashader
    # Plot classes
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.count_cat('class_cat'))
    ckey = dict(GALAXY=(101,236,101), QSO='hotpink', STAR='dodgerblue')
    img = tf.shade(agg, color_key=ckey, how='log')
    export_image(img, 'UMAP-photo-'+label+'-RFclasslabels', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,10)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-RFclasslabels.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    '''
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.Greys), orientation='horizontal', label='Number of sources', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    '''
    # legend for class labels
    g_leg = plt.Line2D((0,0),(0,0), color='lightgreen', marker='', linestyle='', label='Galaxies')
    q_leg = plt.Line2D((0,0),(0,0), color='hotpink', marker='', linestyle='', label='Quasars')
    s_leg = plt.Line2D((0,0),(0,0), color='dodgerblue', marker='', linestyle='', label='Stars')
    leg = plt.legend([g_leg, q_leg, s_leg], ['Galaxies', 'Quasars', 'Stars'], frameon=False)
    leg_texts = leg.get_texts()
    leg_texts[0].set_color('lightgreen')
    leg_texts[1].set_color('hotpink')
    leg_texts[2].set_color('dodgerblue')

    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-RFclasslabels.pdf', bbox_inches='tight')
    plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_PhotoObjs_probs(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader

    # Plot probability mean
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.mean('prob_best'))
    img = tf.shade(agg, cmap=prob_mean_c, how='log')
    export_image(img, 'UMAP-photo-'+label+'-probs-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-probs-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    vmin = agg.data[np.isfinite(agg.data)].min() # isfinite to ignore nans
    vmax = agg.data[np.isfinite(agg.data)].max()
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
    cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
    cax.xaxis.set_minor_formatter(formatter)
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-probs-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot probability STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.std('prob_best'))
    img = tf.shade(agg, cmap=prob_std_c, how='log')
    export_image(img, 'UMAP-photo-'+label+'-probs-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-probs-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a.data)].min() # drop nans
    vmax = a[np.isfinite(a.data)].max()
    #vmin=1.01e-3 # min value is 0.001 so set a little smaller for cbar ticks to be clearer. can't be zero for cbar scale.
    print(vmin, vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    #cax.tick_params(which='major', direction='out', length=4)

    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-probs-std.pdf', bbox_inches='tight')
    plt.close(fig)


    if GQSsplit==True:
        # ------ GALAXIES ------
        # Plot mean prob
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[ (df.class_pred=='GALAXY') ], 'x_'+sup, 'y_'+sup, ds.mean('prob_g'))
        img = tf.shade(agg, cmap=prob_mean_c, how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-probs-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-probs-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
        cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
        cax.xaxis.set_minor_formatter(formatter)
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-probs-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)


        # Plot galaxy std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[ (df.class_pred=='GALAXY') ], 'x_'+sup, 'y_'+sup, ds.std('prob_g'))
        img = tf.shade(agg, cmap=prob_std_c, how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-probs-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-probs-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=1.01e-3
        if sup=='sup':
            vmin=8e-4
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-probs-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)


        # ------ QUASARS ------
        # Plot quasar mean probs
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='QSO'], 'x_'+sup, 'y_'+sup, ds.mean('prob_q'))
        img = tf.shade(agg, cmap=prob_mean_c, how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-probs-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-probs-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
        cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
        cax.xaxis.set_minor_formatter(formatter)
        #cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-probs-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)


        # Plot quasar std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='QSO'], 'x_'+sup, 'y_'+sup, ds.std('prob_q'))
        img = tf.shade(agg, cmap=prob_std_c, how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-probs-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight))  # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-probs-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-probs-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)


        # ------ STARS ------
        # Plot star probs
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='STAR'], 'x_'+sup, 'y_'+sup, ds.mean('prob_s'))
        img = tf.shade(agg, cmap=prob_mean_c, how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-probs-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-probs-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_mean_c), orientation='horizontal', label='Mean probability for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)) # format cbar labels as decimals, not exponents
        cax.xaxis.set_major_formatter(formatter) # apply formatter to major and minor axes
        cax.xaxis.set_minor_formatter(formatter)
        #cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-probs-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot star std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='STAR'], 'x_'+sup, 'y_'+sup, ds.std('prob_s'))
        img = tf.shade(agg, cmap=prob_std_c, how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-probs-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-probs-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin = 1.01e-3
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=prob_std_c), orientation='horizontal', label='Standard deviation of probabilities for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-probs-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_PhotoObjs_resolvedr(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader

    # ------ ALL SOURCES ------

    # Plot resolvedr MEAN
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.mean('resolvedr'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-photo-'+label+'-resolvedr-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    vmin=1.01e-5
    #vmin=7e-6 # keep cbar neat in boundary otherwise labels overlap
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-resolvedr-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot resolvedr STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.std('resolvedr'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-photo-'+label+'-resolvedr-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-resolvedr-std.pdf', bbox_inches='tight')
    plt.close(fig)

    if GQSsplit==True:
        # ------ GALAXIES ------
        # Plot resolvedr MEAN
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='GALAXY'], 'x_'+sup, 'y_'+sup, ds.mean('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-resolvedr-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        vmin=2e-5
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-resolvedr-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot resolvedr STD galaxies
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='GALAXY'], 'x_'+sup, 'y_'+sup, ds.std('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-resolvedr-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        #if sup=='unsup':
        vmin=1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-resolvedr-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # ------ QUASARS ------
        # Plot resolvedr MEAN for low prob quasars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='QSO'], 'x_'+sup, 'y_'+sup, ds.mean('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-resolvedr-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-resolvedr-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot resolvedr STD for low prob quasars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='QSO'], 'x_'+sup, 'y_'+sup, ds.std('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-resolvedr-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        #print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-resolvedr-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)


        # ------ STARS ------
        # Plot resolvedr MEAN for low prob stars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='STAR'], 'x_'+sup, 'y_'+sup, ds.mean('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-resolvedr-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean resolvedr parameter for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-resolvedr-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot resolvedr STD for low prob stars
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df[df.class_pred=='STAR'], 'x_'+sup, 'y_'+sup, ds.std('resolvedr'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-resolvedr-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-resolvedr-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of resolvedr parameter for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-resolvedr-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_PhotoObjs_uz(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader

    # ------ ALL SOURCES ------

    # Plot uz mean
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.mean('SDSS_si'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-photo-'+label+'-uz-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-uz-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    vmin=1.01e-5
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-uz-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot uz STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.std('SDSS_si'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-photo-'+label+'-uz-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-uz-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    vmin=1.01e-5 # keep fig neat
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| magnitude for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-uz-std.pdf', bbox_inches='tight')
    plt.close(fig)


    if GQSsplit==True:
        # ----- GALAXIES -----

        # Plot uz MEAN
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
        agg = cvs.points(df[df.class_pred == 'GALAXY'], 'x_'+sup, 'y_'+sup, ds.mean('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-uz-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-uz-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmax=15
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-uz-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz STD
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
        agg = cvs.points(df[df.class_pred == 'GALAXY'], 'x_'+sup, 'y_'+sup, ds.std('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-uz-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-uz-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        vmin=1.01e-5
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-uz-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- QUASARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
        agg = cvs.points(df[df.class_pred == 'QSO'], 'x_'+sup, 'y_'+sup, ds.mean('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-uz-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-uz-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        vmax=9.9
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-uz-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
        agg = cvs.points(df[df.class_pred == 'QSO'], 'x_'+sup, 'y_'+sup, ds.std('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-uz-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-uz-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        print(vmin,vmax)
        vmin=1.01e-5
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-uz-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- STARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
        agg = cvs.points(df[df.class_pred == 'STAR'], 'x_'+sup, 'y_'+sup, ds.mean('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-uz-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-uz-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        if sup=='unsup':
            vmin=1.01e-5
        vmax=9.9
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |PSF u - PSF z| magnitude for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-uz-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_si'] = np.sqrt( (df['psf_u'] - df['psf_z'])**2 )
        agg = cvs.points(df[df.class_pred == 'STAR'], 'x_'+sup, 'y_'+sup, ds.std('SDSS_si'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-uz-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-uz-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        if sup=='unsup':
            vmin=1.01e-5
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |PSF u - PSF z| for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-uz-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_umap_ds_PhotoObjs_w1w2(df, sup, label='unknown', GQSsplit=False):
    # Plotting: use datashader

    # ------ ALL SOURCES ------

    # Plot uz mean
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.mean('SDSS_wi'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-photo-'+label+'-w1w2-mean', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-w1w2-mean.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    if sup=='unsup':
        vmin=1.01e-3
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-w1w2-mean.pdf', bbox_inches='tight')
    plt.close(fig)



    # Plot uz STD
    cvs = ds.Canvas(plot_width=1000, plot_height=1000)
    df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
    agg = cvs.points(df, 'x_'+sup, 'y_'+sup, ds.std('SDSS_wi'))
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
    export_image(img, 'UMAP-photo-'+label+'-w1w2-std', fmt='.png', background='black')

    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-photo-'+label+'-w1w2-std.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # create new axis below main axis for colourbar
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
    # get min and max values from the data binned by datashader to use as limits for the colourbar
    a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
    vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
    vmax = a[np.isfinite(a)].max()
    #vmax=2
    #vmin=7e-6 # keep fig neat
    print(vmin,vmax)
    cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| magnitude for predicted sources (unclassified dataset)', pad=0.01, cax=cax)
    cax.tick_params(which='both', labelbottom='off')
    fig.tight_layout()
    fig.savefig('UMAP-photo-'+label+'-w1w2-std.pdf', bbox_inches='tight')
    plt.close(fig)


    if GQSsplit==True:
        # ----- GALAXIES -----

        # Plot uz MEAN
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
        agg = cvs.points(df[df.class_pred == 'GALAXY'], 'x_'+sup, 'y_'+sup, ds.mean('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-w1w2-mean-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-w1w2-mean-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=1.01e-3
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-w1w2-mean-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz STD
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
        agg = cvs.points(df[df.class_pred == 'GALAXY'], 'x_'+sup, 'y_'+sup, ds.std('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_g, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-w1w2-std-galaxies', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-w1w2-std-galaxies.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        #vmin = 8e-4
        #vmax = 2
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| for predicted galaxies (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-w1w2-std-galaxies.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- QUASARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
        agg = cvs.points(df[df.class_pred == 'QSO'], 'x_'+sup, 'y_'+sup, ds.mean('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-w1w2-mean-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-w1w2-mean-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        vmin=1.01e-3
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-w1w2-mean-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
        agg = cvs.points(df[df.class_pred == 'QSO'], 'x_'+sup, 'y_'+sup, ds.std('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_q, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-w1w2-std-quasars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-w1w2-std-quasars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        #vmin=8e-4
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| for predicted quasars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-w1w2-std-quasars.pdf', bbox_inches='tight')
        plt.close(fig)



        # ----- STARS -----

        # Plot uz mean
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
        agg = cvs.points(df[df.class_pred == 'STAR'], 'x_'+sup, 'y_'+sup, ds.mean('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-w1w2-mean-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-w1w2-mean-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        if sup=='unsup':
            vmin=1.01e-3
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Mean |w1 - w2| magnitude for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-w1w2-mean-stars.pdf', bbox_inches='tight')
        plt.close(fig)



        # Plot uz std
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        df['SDSS_wi'] = np.sqrt( (df['w1'] - df['w2'])**2 )
        agg = cvs.points(df[df.class_pred == 'STAR'], 'x_'+sup, 'y_'+sup, ds.std('SDSS_wi'))
        img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how='log')
        #img = tf.dynspread(img, threshold=threshold_s, max_px=max_px, shape='square', how='over')
        export_image(img, 'UMAP-photo-'+label+'-w1w2-std-stars', fmt='.png', background='black')

        # generate figure with png created and append colourbar axis
        fig = plt.figure(figsize=(10,fheight)) # y axis larger to fit cbar in
        img = mpimg.imread('UMAP-photo-'+label+'-w1w2-std-stars.png')
        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
        # create new axis below main axis for colourbar
        ax_divider = make_axes_locatable(plt.gca())
        cax = ax_divider.append_axes("bottom", size="3%", pad="1%")
        # get min and max values from the data binned by datashader to use as limits for the colourbar
        a = agg.data[np.nonzero(agg.data)] # remove zeros to stop log colour scale going wrong
        vmin = a[np.isfinite(a)].min() # isfinite to ignore nans
        vmax = a[np.isfinite(a)].max()
        #vmax = 2
        print(vmin,vmax)
        cbar = mpl.pyplot.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=mpl.cm.YlOrBr), orientation='horizontal', label='Standard deviation of |w1 - w2| for predicted stars (unclassified dataset)', pad=0.01, cax=cax)
        cax.tick_params(which='both', labelbottom='off')
        fig.tight_layout()
        fig.savefig('UMAP-photo-'+label+'-w1w2-std-stars.pdf', bbox_inches='tight')
        plt.close(fig)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#-------------------------------------------------
# Main code
#-------------------------------------------------






if __name__ == "__main__":
    # This is an intensive piece of code (can take 2 days to do full run through), UMAP takes a bit of time on over a million data points.
    # Keeping track of labels for supervised/unsupervised runs with different datasets almost always leads to bugs when playing around and doing tests by hand
    # This is the main reason I have not broken it down into parts so the full workflow can be easily reproduced without errors

    # I wouldn't jump in and run this whole code, do it bit by bit. Results are saved to disk as you go to save time when getting to plotting.

    # plot parameters
    mpl.rcParams.update({'font.size': 10})
    mpl.rcParams.update({'figure.dpi': 100})
    fheight = 10 # previously had to change to fit colorbar but now leave as 10x10 square.
    #data shader hyperparms - for increasing size of pixels in low density areas so you can see them
    threshold_g = 0.92 # need different params for each class strangely else it doesn't spread the pixels :/
    threshold_q = 0.90 # these numbers work well for the spec test dataset of 1.5 million sources
    threshold_s = 0.85
    max_px = 2 # more than enough, 3 makes pixels too big

    prob_mean_c = mpl.cm.YlOrBr
    prob_std_c = mpl.cm.YlOrBr


    #colors_g = [(1,1,1), (144/255,238/255,144/255)] lightgreen
    colors_g = [(1,1,1), (101/255,236/255,101/255)] # alexgreen (better on white and black backgrounds)
    cmap_g = make_cmap(colors_g)
    colors_q = [(1,1,1), (255/255,105/255,180/255)] # hotpink
    cmap_q = make_cmap(colors_q)
    colors_s = [(1,1,1), (30/255,144/255,255/255)] # dodgerblue
    cmap_s = make_cmap(colors_s)

    # Input data
    # select features
    psf = ['psf_u', 'psf_g', 'psf_r', 'psf_i', 'psf_z']
    wise = ['w1' ,'w2', 'w3', 'w4']
    feature_columns = psf + wise + ['resolvedr']

    label='spec-halftrain'


    dfspec = load_obj('df_spec_classprobs')
    #dfphoto = load_obj(sys.argv[1])
    print(dfspec.shape)
    #df = df[0::1000] # downsample data for a speedy test run?

    # prepare for umap
    dfspec['class_i'] = -1 # umap requires classes as integers. -1 means no class label available (semi-sup)
    dfspec.loc[dfspec['class']=='GALAXY', 'class_i'] = 1
    dfspec.loc[dfspec['class']=='QSO', 'class_i'] = 2
    dfspec.loc[dfspec['class']=='STAR', 'class_i'] = 3
    dfspec['class_cat'] = dfspec['class'].astype('category')

    # run umap. This takes an hour on the full dataset.
    for supervised, suplab in zip([True, False], ['sup','unsup']):
        dfspec_umap = run_umap_spec(dfspec, feature_columns, label=label, supervised=supervised, sample_train=True)
        dfspec_umap['prob_best'] = dfspec_umap[['prob_g', 'prob_q', 'prob_s']].max(axis=1)
        save_obj(dfspec_umap, 'dfspec-'+label+'-'+suplab) # save for plotting later
        #dfspec_umap = load_obj('dfspec'+label)



    # make plots of spec data with UMAP+DataShader (speedy, so keep outside forloop)

    # plot classes
    for suplab in ['sup','unsup']:
        dfspec_umap = load_obj('dfspec-'+label+'-'+suplab)
        plot_umap_ds_SpecObjs_classes(dfspec_umap, sup=suplab, label=label+'-'+suplab)

    # To get plots broken down by class in the following plot functions, e.g. only plot galaxies in UMAP, set GQSsplit=True. Beware this generates significantly more plots, though could be useful if you want to see the breakdown and distribution of each class individually.

    # plot probabilities
    for suplab in ['sup','unsup']:
        dfspec_umap = load_obj('dfspec-'+label+'-'+suplab)
        plot_umap_ds_SpecObjs_probs(dfspec_umap, sup=suplab, label=label+'-'+suplab, GQSsplit=False)

    # plot resolvedr for
    for suplab in ['sup','unsup']:
        dfspec_umap = load_obj('dfspec-'+label+'-'+suplab)
        plot_umap_ds_SpecObjs_resolvedr(dfspec_umap, sup=suplab, label=label+'-'+suplab, GQSsplit=False)

    # plot |u-z| parameter for UMAP+DataShader
    for suplab in ['sup','unsup']:
        dfspec_umap = load_obj('dfspec-'+label+'-'+suplab)
        plot_umap_ds_SpecObjs_uz(dfspec_umap, sup=suplab, label=label+'-'+suplab, GQSsplit=False)

    # plot |u-z| parameter for UMAP+DataShader
    for suplab in ['sup','unsup']:
        dfspec_umap = load_obj('dfspec-'+label+'-'+suplab)
        plot_umap_ds_SpecObjs_w1w2(dfspec_umap, sup=suplab, label=label+'-'+suplab, GQSsplit=False)


    #exit()




    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    # Now transform photometric data into 2-D space using these supervised and unsupervised UMAP models
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    # - - - Only want to do plotting from pre-existing 'df_photo_umapfromspec' ? Comment out this first section
    #'''
    # - - -

    # Load data
    dfphoto = load_obj('SDSS_allphoto_111M.csv_classified')
    # Load previous models from supervised and unsupervised runs
    umap_model_sup = load_obj('umap_model_supspec-halftrain')
    umap_model_unsup = load_obj('umap_model_unsupspec-halftrain')

    dfphoto_split = np.array_split(dfphoto, 5) # need to chunk it up else it seg faults with all 111 million at once :/
    u_photo_sup=[]
    u_photo_unsup=[]
    # kinda slow part here, lots of data, takes a good ~6-12 hours
    for d, idx in zip(dfphoto_split, range(1,6)):
        print('Transforming dfphoto part {0}/5...'.format(idx))
        features = d[[*feature_columns]]
        # supervised model
        u_sup = umap_model_sup.transform(features)
        u_photo_sup.append(u_sup)
        # unsupervised model
        u_unsup = umap_model_unsup.transform(features)
        u_photo_unsup.append(u_unsup)

    # concatenate arrays. loop over two models. keeping automated track of labels is main reason for this.
    print('Cleaning up dataframes...') # pretty quick from here
    for u_photo, suplab in zip([u_photo_sup, u_photo_unsup], ['sup', 'unsup']):
        u_photo = np.concatenate((u_photo[0], u_photo[1], u_photo[2], u_photo[3], u_photo[4])) # join 5 parts
        u_photo = pd.DataFrame(u_photo, columns=['x_'+suplab, 'y_'+suplab], index=dfphoto.index) # label cols for sup/unsup
        # join umap projection to original df
        dfphoto = dfphoto.join(u_photo, how='left')

    # create catagorical class for datashader colours (bc it wont accept other formats)
    dfphoto['class_cat'] = dfphoto['class_pred'].astype('category')
    dfphoto['prob_best'] = dfphoto[['prob_g', 'prob_q', 'prob_s']].max(axis=1)
    # save dfs for quick plotting later
    save_obj(dfphoto, 'df_photo_umapfromspec')

    # - - - End of processing section
    #'''
    # - - - Plotting below



    print('Plotting all photo objects...')

    # load photo df with umap columns added (can comment out above sections if df already exists with umap columns)
    dfphoto = load_obj('df_photo_umapfromspec')

    dfphoto['prob_best'] = dfphoto[['prob_g', 'prob_q', 'prob_s']].max(axis=1)
    dfphoto['resolvedr'] = (np.sqrt((dfphoto['cmod_r'] - dfphoto['psf_r'])**2))


    # Plot classes for UMAP+DataShader
    for suplab in ['sup','unsup']:
        plot_umap_ds_PhotoObjs_classes(dfphoto, sup=suplab, label=label+'-'+suplab)

    # To get plots broken down by class, e.g. only plot galaxies in UMAP, set GQS-split=True. Beware this generates significantly more plots, though could be useful if you want to see the breakdown and distribution of each class individually.

    for suplab in ['sup','unsup']:
        plot_umap_ds_PhotoObjs_probs(dfphoto, sup=suplab, label=label+'-'+suplab, GQSsplit=False)

    # plot resolved parameter for UMAP+DataShader
    for suplab in ['sup','unsup']:
        plot_umap_ds_PhotoObjs_resolvedr(dfphoto, sup=suplab, label=label+'-'+suplab, GQSsplit=False)

    # plot |u-z| parameter for UMAP+DataShader
    for suplab in ['sup','unsup']:
        plot_umap_ds_PhotoObjs_uz(dfphoto, sup=suplab, label=label+'-'+suplab, GQSsplit=False)

    # plot |w1-w2| parameter for UMAP+DataShader
    for suplab in ['sup','unsup']:
        plot_umap_ds_PhotoObjs_w1w2(dfphoto, sup=suplab, label=label+'-'+suplab, GQSsplit=False)


    #exit()


    # Add spec data to photo data for UMAP tests on semi-sup with spec to help

    dfphoto = load_obj('df_photo_umapfromspec')
    #dfphoto['prob_best'] = dfphoto[['prob_g', 'prob_q', 'prob_s']].max(axis=1)
    #dfphoto['resolvedr'] = (np.sqrt((dfphoto['cmod_r'] - dfphoto['psf_r'])**2))

    dfphoto = dfphoto.sample(frac=0.1) # fraction of 111 million
    dfphoto['class_i'] = -1


    # plot 11M photo unsupervised
    #dfphoto_tmp = run_umap_photo(dfphoto, feature_columns, label='withspec', labels=False)
    #save_obj(dfphoto_tmp, 'dfphoto-umap-unsup')
    dfphoto_tmp = load_obj('dfphoto-umap-unsup')
    plot_umap_ds_PhotoObjs_classes(dfphoto_tmp, sup='', label='')
    #exit()

    # add spec data?
    dfspec = load_obj('df_spec_classprobs')
    dfspec = dfspec.drop(columns=['class_pred']) # don't need for this and will rename later
    dfspec['class_i'] = -1 # means no label in semi-sup umap
    dfspec.loc[dfspec['class'] == 'GALAXY', 'class_i'] = 0
    dfspec.loc[dfspec['class'] == 'QSO', 'class_i'] = 1
    dfspec.loc[dfspec['class'] == 'STAR', 'class_i'] = 2
    #df['class_cat'] = df['class_str'].astype('category')
    dfspec.rename(columns={'class': 'class_pred'}, inplace=True) # else not using real labels in umap plot
    # append spec data to photo
    dfphoto = dfphoto.append(dfspec, ignore_index=True)

    # don't change dfphoto now for the following runs. use dfphoto_tmp

    # plot 11M photo + all spec unsupervised
    dfphoto_tmp = run_umap_photo(dfphoto, feature_columns, label='withspec', labels=False)
    save_obj(dfphoto_tmp, 'dfphoto-umap-unsup-withspec')
    dfphoto_tmp = load_obj('dfphoto-umap-unsup-withspec')
    plot_umap_ds_PhotoObjs_classes(dfphoto_tmp, sup='', label='withspec')

    # plot 11M photo + all spec semi supervised
    print('Doing semi-supervised UMAP run with spec labels included...')
    dfphoto_tmp = run_umap_photo(dfphoto, feature_columns, label='semisupspec', labels=True)
    save_obj(dfphoto_tmp, 'dfphoto-umap-unsup-semisupspec')
    dfphoto_tmp = load_obj('dfphoto-umap-unsup-semisupspec')
    plot_umap_ds_PhotoObjs_classes(dfphoto_tmp, sup='', label='semisupspec')


    # to fix a problem i had when i lost the class_pred column...
    #dfphoto.loc[dfphoto['class_i'] == 0, 'class_pred'] = 'GALAXY'
    #dfphoto.loc[dfphoto['class_i'] == 1, 'class_pred'] = 'QSO'
    #dfphoto.loc[dfphoto['class_i'] == 2, 'class_pred'] = 'STAR'





    # End
