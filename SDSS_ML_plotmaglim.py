# Written by Alex Clarke - informationcake.github.com
# Creates magnitude limited plots

# Pre-requisits: Run SDSS_ML.py to obtain the following .pkl files for plot_metrics_trainlimit:
# metrics_df_185mag.pkl
# metrics_df_19mag.pkl
# metrics_df_195mag.pkl
# metrics_df_20mag.pkl
# metrics_df_205mag.pkl
# metrics_df_21mag.pkl

# Version numbers
# python: 3.6.1
# pandas: 0.25.0
# numpy: 1.17.0
# scipy: 1.3.1
# matplotlib: 3.1.1
# sklearn: 0.20.0

# Import functions
import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg',warn=False, force=True) #set MPL backend.
import matplotlib.pyplot as plt
import pickle #save/load python data objects (dictionaries/arrays)

# list of functions:
# lsave_obj load_obj
# plot_metrics_trainlimit
# plot_trainlimit



# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#Loading/saving python data objects
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_metrics_trainlimit(rev=False):
    # make plots of metrics for various runs where the training has been restricted to a magnitude limit
    if rev==False:
        try:
            m185 = load_obj('metrics_df_185m')
            m19 = load_obj('metrics_df_19m')
            m195 = load_obj('metrics_df_195m')
            m20 = load_obj('metrics_df_20m')
            m205 = load_obj('metrics_df_205m')
            m21 = load_obj('metrics_df_21m')
        except:
            print('Missing files, did you run SDSS_ML_analysis.py plot_metric_curves to save the data to disk?')

    if rev==True:
        try:
            m185 = load_obj('metrics_df_185mag_rev')
            m19 = load_obj('metrics_df_19mag_rev')
            m195 = load_obj('metrics_df_195mag_rev')
            m20 = load_obj('metrics_df_20mag_rev')
            m205 = load_obj('metrics_df_205mag_rev')
            m21 = load_obj('metrics_df_21mag_rev')
        except:
            print('Missing files, did you run SDSS_ML_analysis.py plot_metric_curves to save the data to disk?')

    m_all = load_obj('metrics_df_allmag')

    
    # metrics = {'x':x_tmp, 'g':g, 's':s, 'q':q, 'pg':pg, 'rg':rg, 'f1g':f1g, 'pgerr':pgerr, 'rgerr':rgerr, 'f1gerr':f1gerr, 'pq':pq, 'rq':rq, 'f1q':f1q, 'pqerr':pqerr, 'rqerr':rqerr, 'f1qerr':f1qerr, 'ps':ps, 'rs':rs, 'f1s':f1s, 'pserr':pserr, 'rserr':rserr, 'f1serr':f1serr}
    # probs = {'g_probs':g_probs, 's_probs':s_probs, 'q_probs':q_probs}

    x_maglims = {18.5:m185, 19:m19, 19.5:m195, 20:m20, 20.5:m205, 21:m21, 'all':m_all}
    #x_maglims = {18.5:m185}
    colors = mpl.cm.rainbow(np.linspace(0, 1, len(x_maglims)))

    f, axs = plt.subplots(3, 3, figsize=(10,7), sharey=True, sharex=False)

    # loop over metrics
    for ax_i, metric in zip([0,1,2], ['p','r','f1']):
        ## ------ ------ Galaxies ------ ------
        plt.sca(axs[ax_i, 0])
        for key, c in zip(x_maglims, colors):
            if key=='all':
                plt.plot(x_maglims[key]['x'], x_maglims[key][metric+'g'], label=key, color='black', linewidth=0.5)
            if key!='all':
                plt.plot(x_maglims[key]['x'], x_maglims[key][metric+'g'], label=key, color=c, linewidth=0.5)
        plt.vlines(x=list(x_maglims.keys())[:-1], ymin=0, ymax=1, colors=colors, ls='--', linewidth=0.5)
        plt.xticks(np.arange(15,26,2)) # custom ticks to prevent overlapping at edges with wspace=0
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        plt.xlim(14,26)
        plt.minorticks_on()
        if metric=='p':
            plt.ylabel('Precision')
        if metric=='r':
            plt.ylabel('Recall')
        if metric=='f1':
            plt.ylabel('F1 score')
        if ax_i==2:
            plt.xticks(np.arange(15,26,2)) # custom ticks to prevent overlapping at edges with wspace=0
            plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
            plt.xlabel('PSF r magnitude for galaxies')
        if ax_i==1:
            plt.legend(frameon=False, fontsize=8, bbox_to_anchor=(0.35, 0.65))

        #plt.fill_between(x, gp+gp_std, gp-gp_std, color=galaxy_c, step='mid', linewidth=0, alpha=alpha)

        ## ------ ------ Quasars ------ ------
        plt.sca(axs[ax_i, 1])
        for key, c in zip(x_maglims, colors):
            if key=='all':
                plt.plot(x_maglims[key]['x'], x_maglims[key][metric+'q'], label=key, color='black', linewidth=0.5)
            if key!='all':
                plt.plot(x_maglims[key]['x'], x_maglims[key][metric+'q'], label=key, color=c, linewidth=0.5)
        plt.vlines(x=list(x_maglims.keys())[:-1], ymin=0, ymax=1, colors=colors, ls='--', linewidth=0.5)
        plt.xticks(np.arange(15,26,2)) # custom ticks to prevent overlapping at edges with wspace=0
        plt.xlim(14,26)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        plt.minorticks_on()
        if ax_i==2:
            plt.xticks(np.arange(15,26,2)) # custom ticks to prevent overlapping at edges with wspace=0
            plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
            plt.xlabel('PSF r magnitude for quasars')

        ## ------ ------ Stars ------ ------
        plt.sca(axs[ax_i, 2])
        for key, c in zip(x_maglims, colors):
            if key=='all':
                plt.plot(x_maglims[key]['x'], x_maglims[key][metric+'s'], label=key, color='black', linewidth=0.5)
            if key!='all':
                plt.plot(x_maglims[key]['x'], x_maglims[key][metric+'s'], label=key, color=c, linewidth=0.5)
        plt.vlines(x=list(x_maglims.keys())[:-1], ymin=0, ymax=1, colors=colors, ls='--', linewidth=0.5)
        plt.xticks(np.arange(9,26,2)) # custom ticks to prevent overlapping at edges with wspace=0
        plt.xlim(8,26)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=True, right=True)
        plt.minorticks_on()
        if ax_i==2:
            plt.xticks(np.arange(9,26,2)) # custom ticks to prevent overlapping at edges with wspace=0
            plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
            plt.xlabel('PSF r magnitude for stars')

    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    plt.savefig('metrics-train-maglim.pdf', bbox_inches='tight', dpi=400)






# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def plot_trainlimit(rev=False):
    # make plots of metrics for various runs where the training has been restricted to a magnitude limit
    if rev==False:
        try:
            m185 = load_obj('metrics_df_185m')
            m19 = load_obj('metrics_df_19m')
            m195 = load_obj('metrics_df_195m')
            m20 = load_obj('metrics_df_20m')
            m205 = load_obj('metrics_df_205m')
            m21 = load_obj('metrics_df_21m')
        except:
            print('Missing files, did you run SDSS_ML_analysis.py plot_metric_curves to save the data to disk?')

    if rev==True:
        try:
            m185 = load_obj('metrics_df_185mag_rev')
            m19 = load_obj('metrics_df_19mag_rev')
            m195 = load_obj('metrics_df_195mag_rev')
            m20 = load_obj('metrics_df_20mag_rev')
            m205 = load_obj('metrics_df_205mag_rev')
            m21 = load_obj('metrics_df_21mag_rev')
        except:
            print('Missing files, did you run SDSS_ML_analysis.py plot_metric_curves to save the data to disk?')

    m_all = load_obj('metrics_df_allmag')

    # metrics = {'x':x_tmp, 'pg':pg, 'rg':rg, 'f1g':f1g, 'pgerr':pgerr, 'rgerr':rgerr, 'f1gerr':f1gerr, 'pq':pq, 'rq':rq, 'f1q':f1q, 'pqerr':pqerr, 'rqerr':rqerr, 'f1qerr':f1qerr, 'ps':ps, 'rs':rs, 'f1s':f1s, 'pserr':pserr, 'rserr':rserr, 'f1serr':f1serr}

    x_maglims = {18.5:m185, 19:m19, 19.5:m195, 20:m20, 20.5:m205, 21:m21, 'all':m_all}
    colors = mpl.cm.rainbow(np.linspace(0, 1, len(x_maglims)))

    # get first time the f1 score drops below a certain value, with the condition that r mag is greater than the training limit:
    #for key in list(x_maglims.keys())[:-1]:
    #    x_maglims[key]['ratio'] = x_maglims[key]['f1g']/x_maglims['all']['f1g'] # append ratio to df

    f, axs = plt.subplots(1, 3, figsize=(10,4), sharey=True, sharex=False)
    # loop over source type
    for ax_i, type in zip([0,1,2], ['g','q','s']):
        plt.sca(axs[ax_i])
        f1lim=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        for f1lim, c in zip(f1lim, colors): # loop over f1 limits. f1-lim is a fraction of f1 without training limit applied.
            a=[]
            for key in list(x_maglims.keys())[:-1]:
                try:
                    if rev==False:
                        a.append( x_maglims[key]['x'][ (x_maglims[key]['x'] > float(key)) & (x_maglims[key]['f1'+type]/x_maglims['all']['f1'+type] < f1lim) ].iloc[0] )
                    if rev==True:
                        a.append( x_maglims[key]['x'][ (x_maglims[key]['x'] < float(key)) & (x_maglims[key]['f1'+type]/x_maglims['all']['f1'+type] < f1lim) ].iloc[-1] )
                except:
                    a.append(np.nan) # doesn't drop below limit
            plt.plot(list(x_maglims.keys())[:-1], a, ls='dashed', marker='x', linewidth=1, label=f1lim)
        # plot y=x line for guide
        plt.plot(np.arange(17, 22, 0.1), np.arange(17, 22, 0.1), linewidth=0.2, color='black', label='y = x')

        if ax_i==0:
            plt.xlabel('Training r magnitude limit for galaxies')
            plt.ylabel('r magnitude where fraction of F1-score drops below limit')
        if ax_i==1:
            plt.xlabel('Training r magnitude limit for quasars')
            plt.legend(frameon=False, title='Fraction of unlimited F1-score')
            leg = plt.legend(frameon=False, title='Fraction of F1-score \nwithout training limit')
            leg._legend_box.align = "left"
        if ax_i==2:
            plt.xlabel('Training r magnitude limit for stars')

        plt.tick_params(axis='x', which='both', top=True, bottom=True)
        plt.tick_params(axis='y', which='both', left=True, right=True)
        plt.minorticks_on()
        if rev==False:
            plt.xlim(18.3, 21.2)
            plt.ylim(18,25)
        #if rev==True:

    f.tight_layout()
    f.subplots_adjust(wspace=0, hspace=0) # Must come after tight_layout to work!
    plt.savefig('train_lim.pdf')












#-------------------------------------------------
# Main
#-------------------------------------------------






if __name__ == "__main__": #so you can import this code and run by hand if desired

    print('_'*50)
    # Set defaults for all plots
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
    correct_capsize = 4
    missed_capsize = 8
    linewidth = 1
    rotation = 0

    # Make plots investigating the magnitude limit cut
    # This requires running SDSS_ML.py with a mag limit, then running metric_curves() rom SDSS_ML_analysis.py to get the datafiles
    # In SDSS_ML.py, in the function prepare_data() set mag_split=True and ttsplit=False in prepare_data()
    # Iterate this setting mag_lim each time
    # Filenames are hardcoded in the plotting functions in this code.
    # I should probably rewrite this to be more automated. At the moment it is quite an involved process

    plot_metrics_trainlimit(rev=False)

    plot_trainlimit()





    # end
