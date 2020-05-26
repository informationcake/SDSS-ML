# Written by Alex Clarke - https://github.com/informationcake/SDSS-ML
# Train and test a machine learning model on SDSS and WISE spectroscopically confirmed galaxies, quasars and stars

# Version numbers
# python: 3.6.1
# pandas: 0.25.0
# numpy: 1.17.0
# scipy: 1.3.1
# matplotlib: 3.1.1
# sklearn: 0.20.0

import os, sys, glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
#mpl.use('TKAgg',warn=False, force=True) #set MPL backend.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle #save/load python data objects (dictionaries/arrays)
import time
import itertools
from textwrap import wrap #Long figure titles
import multiprocessing
#from memory_profiler import profile #profile memory

#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# List of functions:
# save_obj
# load_obj
# prepare_data
# RF_fit
# RF_classify
# metrics
# transform_features
# drop_duplicates
# train_vs_f1score
# crossvalidation_hyperparms
# load_and_clean_data



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#Loading/saving python data objects
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# Take input data, prepare for supervised ML or TSNE
def prepare_data(data, feature_columns, train_percent=0.5, ttsplit=True, mag_split=False, mag_lim=18.5, ttbelow=True, scale=False, verbose=False, newsources=False):
    print('Preparing data... Its shape is: {0}'.format(data.shape))
    #test/train split using sklearn function
    if ttsplit==True and mag_split==False:
        all_features = data[[*feature_columns]]
        if newsources==False:
            all_classes = data['class']
        if newsources==True:
            all_classes = data['class_pred']
        if scale==True:
            print('Scaling features...')
            #all_features = preprocessing.scale(all_features)
            all_features = preprocessing.normalize(all_features)
            all_features = pd.DataFrame(all_features)
        features_train, features_test, classes_train, classes_test = train_test_split(all_features, all_classes, train_size=train_percent, random_state=0, stratify=all_classes)
        class_names = np.unique(all_classes)
        feature_names = list(all_features)
        #print(isinstance(classes_train, pd.DataFrame))
        if verbose==True: print('feature names are: ', str(feature_names))
        return {'features_train':features_train, 'features_test':features_test, 'classes_train':classes_train, 'classes_test':classes_test, 'class_names':class_names, 'feature_names':feature_names} #return dictionary. data within dictionary are DataFrames.

    #test/train split only below a PSF magnitude limit in r band
    if ttsplit==True and mag_split==True:
        if ttbelow==True:
            print('Imposing magnitude cut of cmod_r={0}, performing test/train split below this...'.format(mag_lim))
            data_tt = data[data.psf_r < mag_lim] # split out brighter fraction of data before tt split
            data_new = data[data.psf_r > mag_lim] # set aside fraction of fainter new sources (e.g. simulating deeper data)
        if ttbelow==False:
            print('Imposing magnitude cut of cmod_r={0}, performing test/train split above this...'.format(mag_lim))
            data_tt = data[data.psf_r > mag_lim] # split out fainter fraction of data before tt split
            data_new = data[data.psf_r < mag_lim] # set aside fraction of brighter sources
        all_features = data_tt[[*feature_columns]]
        all_classes = data_tt['class']
        print('Number of sources not in tt-split: {0}'.format(len(data_new)))
        # do tt split
        features_train, features_test, classes_train, classes_test = train_test_split(all_features, all_classes, train_size=train_percent, random_state=0, stratify=all_classes)
        # append the data_new deeper sources not included in tt split to the arrays
        features_test = features_test.append( data_new[[*feature_columns]] )
        classes_test = classes_test.append( data_new['class'] )
        print('Training on \n{0}'.format(classes_train.value_counts()))
        print('Testing on \n{0}'.format(classes_test.value_counts()))
        # get names as strings
        class_names = np.unique(data['class'])
        feature_names = list(data[[*feature_columns]])
        if verbose==True: print('feature names are: ', str(feature_names))

        return {'features_train':features_train, 'features_test':features_test, 'classes_train':classes_train, 'classes_test':classes_test, 'class_names':class_names, 'feature_names':feature_names} #return dictionary


    #no test/train split, just return data in format for e.g. tsne or clustering
    if ttsplit==False:
        all_features = data[[*feature_columns]]
        all_classes = data['class']
        class_names = np.unique(data['class'])
        return {'all_features':all_features, 'all_classes':all_classes, 'class_names':class_names} #return dictionary






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#Fit a Random Forest.
def RF_fit(data, n_estimators, n_jobs=-1):
    print('Fitting a random forest model to the data...')
    #Prepare classifier with hyper parameters
    rfc=RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators, random_state=0, class_weight='balanced')
    #Set up sklean pipeline to keep everything together
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators, random_state=0, class_weight='balanced')) ])
    #Do the fit
    pipeline.fit(data['features_train'], data['classes_train'])
    return pipeline






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# Classify using fitted Random Forest model
def RF_classify(pipeline, data, n_jobs=-1, ttsplit=True, proba=False):
    if ttsplit==True:
        print('Classifying objects using random forest model...')
        if proba==False:
            classes_pred = pipeline.predict(data['features_test'])
            return classes_pred
        if proba==True:
            classes_pred = pipeline.predict_proba(data['features_test'])
            return classes_pred
    if ttsplit==False: # must have used tsne=True option in prepare_data, implying no test/train split
        print('Classifying objects using random forest model (not used in test/train split)...')
        if proba==False:
            classes_pred = pipeline.predict(data['all_features'])
            return classes_pred
        if proba==True:
            classes_pred = pipeline.predict_proba(data['all_features'])
            return classes_pred






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# Return metrics to assess performance of the model
def metrics(data, classes_pred, ttsplit=True):
    if ttsplit==True:
        report=classification_report(data['classes_test'], classes_pred, target_names=np.unique(data['class_names']), digits=4)
        print(report)
        print(confusion_matrix(data['classes_test'], classes_pred, labels=data['class_names']))
    if ttsplit==False: # must have used tsne=True option in prepare_data, implying no test/train split
        report=classification_report(data['all_classes'], classes_pred, target_names=np.unique(data['class_names']), digits=4)
        print(report)
        print(confusion_matrix(data['all_classes'], classes_pred, labels=data['class_names']))






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






# Add a column to the df which is a 1-D transformation of the 10-D feature space.
def transform_features(df, feature_columns, n_components=1):
    # Use Principal Component Analysis
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit( df[[*feature_columns]] )
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.components_)
    print(pca.singular_values_)

    # Add new column to data frame with features transformed into 1D
    feature_1D = pca.transform( df[[*feature_columns]] )
    #print(feature_1D.shape)
    df['feature_1D'] = feature_1D
    #df['feature_2D_1'] = feature_1D[:,0]
    #df['feature_2D_2'] = feature_1D[:,1]
    print('df now has new column called "feature_1D" for all sources')
    # Return nothing, since appending to df within function






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def drop_duplicates(df):
    # Find duplicate SpecOBJIDs:
    df_dup = df[df.specObjID.duplicated()==True]
    i = df[df.specObjID.duplicated()==True].specObjID
    drop_idxs = []
    # Loop over them, work out maximum match distance, append to list.
    for idx in i:
        m = df[df.specObjID==idx].match_dist
        drop_idxs.append(m.idxmax())
    # Drop entires with maximum match_dist, leaving best matching object.
    df = df.drop(drop_idxs)
    return df






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def train_vs_f1score(df, sampleG=False):
    train_range = [0.001, 0.003, 0.01, 0.06, 0.12, 0.2, 0.4, 0.6, 1.0]
    f1scores=[]
    precisions=[]
    recalls=[]
    f1scores.append(train_range) # append x-axis as first entry to make plotting easier later when loading data from disk.
    print('Looping over these possible train percentages: {0}'.format(train_range))
    # split out half initially. fix test set for all models.
    data_prep_dict_all = prepare_data(df, feature_columns, train_percent=0.5, mag_split=False, verbose=False, ttsplit=True)
    # set up RF pipeline
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=0, class_weight='balanced')) ])
    #pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=0)) ])
    # loop over fractions of this half to test on other half
    for i in train_range:
        print('train percent is: {0}'.format(i))
        if i!=1.0:
        # train test split on the half seclected
            features_train, features_test, classes_train, classes_test = train_test_split(data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], train_size=i, random_state=0, stratify=data_prep_dict_all['classes_train'])
        if i==1.0:
            features_train = data_prep_dict_all['features_train']
            features_test = data_prep_dict_all['features_test']
            classes_train = data_prep_dict_all['classes_train']
            classes_test = data_prep_dict_all['classes_test']

        print('number of sources available for training {0}'.format(len(features_train)))
        if sampleG==True:
            print('sampling galaxies to fix class imbalance...')
            galaxy_features = features_train[classes_train == 'GALAXY']
            quasar_features = features_train[classes_train == 'QSO']
            star_features = features_train[classes_train == 'STAR']
            galaxy_classes = classes_train[classes_train == 'GALAXY']
            quasar_classes = classes_train[classes_train == 'QSO']
            star_classes = classes_train[classes_train == 'STAR']
            # take 20% of galaxies
            galaxy_features = galaxy_features[0::5]
            galaxy_classes = galaxy_classes[0::5]
            # recombine features and classes
            features_train = pd.concat([galaxy_features, quasar_features, star_features])
            classes_train = pd.concat([galaxy_classes, quasar_classes, star_classes])
            print('Training on {0}... G: {1}, Q: {2}, S: {3}'.format(len(features_train), len(galaxy_features), len(quasar_features), len(star_features)))
            # shuffle data... shouldn't (DOESN'T) make any difference...
            p = np.random.permutation(len(features_train))
            features_train = np.array(features_train)[p]
            classes_train = np.array(classes_train)[p]

        if sampleG==False:
            galaxy_classes = classes_train[classes_train == 'GALAXY']
            quasar_classes = classes_train[classes_train == 'QSO']
            star_classes = classes_train[classes_train == 'STAR']
            print('Training on {0}... G: {1}, Q: {2}, S: {3}'.format(len(features_train), len(galaxy_classes), len(quasar_classes), len(star_classes)))

        # fit rf on subset of train data
        pipeline.fit(features_train, classes_train)
        # predict classes of the original 50% not used
        classes_pred = pipeline.predict(data_prep_dict_all['features_test'])
        f1score = f1_score(data_prep_dict_all['classes_test'], classes_pred, average=None)
        precision = precision_score(data_prep_dict_all['classes_test'], classes_pred, average=None)
        recall = recall_score(data_prep_dict_all['classes_test'], classes_pred, average=None)
        print(f1score)
        print(precision)
        print(recall)
        f1scores.append(f1score)
        precisions.append(precision)
        recalls.append(recall)
        print('-'*30)

    print(f1score)
    if sampleG==False:
        save_obj(f1scores, 'train_vs_f1score')
        save_obj(precisions, 'train_vs_precision')
        save_obj(recalls, 'train_vs_recall')
    if sampleG==True:
        save_obj(f1scores, 'train_vs_f1score_sampleG')
        save_obj(precisions, 'train_vs_precision_sampleG')
        save_obj(recalls, 'train_vs_recall_sampleG')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def crossvalidation_hyperparms():
    # Now fix percentage of data to train on:
    train_percent = 0.5

    trees_range = [10, 25, 50, 75, 100, 150, 200, 1000]
    print('Looping over these possible number of trees to use: {0}'.format(trees_range))
    for i in trees_range:
        print('Number of trees is: {0}'.format(i))
        data_prep_dict_all = prepare_data(df, feature_columns, train_percent=train_percent, mag_split=False, verbose=False, ttsplit=False)
        pipeline = RF_fit(data_prep_dict_all, n_estimators=i, n_jobs=-1)
        classes_pred_all = RF_classify(pipeline, data_prep_dict_all)
        metrics(data_prep_dict_all, classes_pred_all)
        print('-'*30)

    data_prep_dict_all = prepare_data(df, feature_columns, train_percent=0.5, mag_split=False, verbose=False, ttsplit=True)

    print('cross-validating...')
    all_scores = []
    for leaf in [1,5, 10, 50, 100, 500]:
        print(leaf)
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_leaf=leaf, random_state=0, class_weight='balanced')
        scores = cross_validate(rfc, data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], scoring='f1_weighted', cv=5, n_jobs=-1, return_train_score=True)
        all_scores.append([leaf, scores])
        print('-'*30)
    save_obj(all_scores, 'cv_scores_leaf')

    print('cross-validating...')
    all_scores = []
    for n_estimators in [20, 50, 100, 200, 500, 1000]:
        print(n_estimators)
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, random_state=0, class_weight='balanced')
        scores = cross_validate(rfc, data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], scoring='f1_weighted', cv=5, n_jobs=-1, return_train_score=True)
        all_scores.append([n_estimators, scores])
        print('-'*30)
    save_obj(all_scores, 'cv_scores_trees')


    print('cross-validating...')
    all_scores = []
    for feat in [2,3,4,5,6]:
        print(feat)
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_features=feat, random_state=0, class_weight='balanced')
        scores = cross_validate(rfc, data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], scoring='f1_weighted', cv=5, n_jobs=-1, return_train_score=True)
        all_scores.append([feat, scores])
        print('-'*30)
    save_obj(all_scores, 'cv_scores_features')

    print('correcting class imbalance')
    # class imbalance fix:
    df_g = df[df['class']=='GALAXY'][0::5]
    df_q = df[df['class']=='QSO']
    df_s = df[df['class']=='STAR']
    df = pd.concat([df_g, df_q, df_s])
    print(df['class'].value_counts())

    print('cross-validating...')
    all_scores = []
    for leaf in [1,5, 10, 50, 100, 500]:
        print(leaf)
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_leaf=leaf, random_state=0, class_weight='balanced')
        scores = cross_validate(rfc, data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], scoring='f1_weighted', cv=5, n_jobs=-1, return_train_score=True)
        all_scores.append([leaf, scores])
        print('-'*30)
    save_obj(all_scores, 'cv_scores_leaf_CI')

    print('cross-validating...')
    all_scores = []
    for n_estimators in [20, 50, 100, 200, 500, 1000]:
        print(n_estimators)
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, random_state=0, class_weight='balanced')
        scores = cross_validate(rfc, data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], scoring='f1_weighted', cv=5, n_jobs=-1, return_train_score=True)
        all_scores.append([n_estimators, scores])
        print('-'*30)
    save_obj(all_scores, 'cv_scores_trees_CI')


    print('cross-validating...')
    all_scores = []
    for feat in [2,3,4,5,6]:
        print(feat)
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_features=feat, random_state=0, class_weight='balanced')
        scores = cross_validate(rfc, data_prep_dict_all['features_train'], data_prep_dict_all['classes_train'], scoring='f1_weighted', cv=5, n_jobs=-1, return_train_score=True)
        all_scores.append([feat, scores])
        print('-'*30)
    save_obj(all_scores, 'cv_scores_features_CI')






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






def load_and_clean_data(file, feature_columns):
    #df = pd.read_csv('SDSS_spec_xmwise.csv')
    df = pd.read_csv(file)
    print(df['class'].value_counts())
    print(len(df))
    df = drop_duplicates(df)
    print('after removing duplicate wise matches:')
    print(df['class'].value_counts())
    print(len(df))

    #Filter the data to remove bad entires
    df=df[(df.cmod_u>0.0) & (df.cmod_g>0.0) & (df.cmod_r>0.0) & (df.cmod_i>0.0) & (df.cmod_z>0.0)] #remove objects where cmodel flux fit has given negative value (118 entires). In practise this is the same as using psf. -9999 values present.
    print('after removing cmod -9999s')
    print(len(df))
    #print(df.zwarning.value_counts())
    df=df[(df.zwarning==0) | (df.zwarning==16)] # | (df.zwarning==4)] #removes 81314 objects with poor quality spectra. Spectra with zWarning equal to zero have no known problems. zWarning==16 (MANY_OUTLIERS) is only ever set in the data taken with the SDSS spectrograph, not the BOSS spectrograph (the SDSS-I, -II and SEGUE-2 surveys). If it is set, that usually indicates a high signal-to-noise spectrum or broad emission lines in a galaxy; that is, MANY_OUTLIERS only rarely signifies a real error.
    print('after removing zwar flags')
    print(len(df))
    #zWarning==16 keeps 4111 objects (4113 excluding the cmod_ conditions above)
    df=df[(df.w1<50) & (df.w2<50) & (df.w3<50) & (df.w4<50)]
    #df=df[(df.w1<50) & (df.w2<50)]
    print('after removing wise 9999s')
    print(len(df))

    #df now Left with 2,457,349 objects. 81432 removed.
    print('After filtering, data frame has shape: {0}'.format(df.shape))
    print('SDSS: \n{0}'.format(df[df.instrument=='SDSS']['class'].value_counts()))
    print('BOSS: \n{0}'.format(df[df.instrument=='BOSS']['class'].value_counts()))
    print('Final count: \n')
    print(df['class'].value_counts())



    #Create columns corrected for galactic extinction
    '''
    df['cmod_u_corr'] = df.cmod_u - df.ext_u
    df['cmod_g_corr'] = df.cmod_g - df.ext_g
    df['cmod_r_corr'] = df.cmod_r - df.ext_r
    df['cmod_i_corr'] = df.cmod_i - df.ext_i
    df['cmod_z_corr'] = df.cmod_z - df.ext_z
    df['psf_u_corr'] = df.psf_u - df.ext_u
    df['psf_g_corr'] = df.psf_g - df.ext_g
    df['psf_r_corr'] = df.psf_r - df.ext_r
    df['psf_i_corr'] = df.psf_i - df.ext_i
    df['psf_z_corr'] = df.psf_z - df.ext_z
    '''
    #df['resolvedu'] = np.sqrt((df.psf_u_corr - df.cmod_u_corr)**2)
    #df['resolvedg'] = np.sqrt((df.psf_g_corr - df.cmod_g_corr)**2)
    df['resolvedr'] = np.sqrt((df.psf_r - df.cmod_r)**2)
    #df['resolvedi'] = np.sqrt((df.psf_i_corr - df.cmod_i_corr)**2)
    #df['resolvedz'] = np.sqrt((df.psf_z_corr - df.cmod_z_corr)**2)
    #df_zwar4 = df[df.zwarning==4]
    #print(df.nlargest(20, 'ext_r').ext_r)
    # extinction correction checks
    #print('sources with ext greater than 1: {0}'.format(len(df[df.ext_r > 1])))
    #print('sources with ext greater than 5: {0}'.format(len(df[df.ext_r > 5])))
    #print('Median and STD of extinction correction')
    #print(df.ext_u.median(), (df.ext_g.median()), (df.ext_r.median()), (df.ext_i.median()), (df.ext_z.median()))
    #print(df.ext_u.std(), (df.ext_g.std()), (df.ext_r.std()), (df.ext_i.std()), (df.ext_z.std()))
    #df = df[df.ext_r < 1]
    #print(df.ext_u.median(), (df.ext_g.median()), (df.ext_r.median()), (df.ext_i.median()), (df.ext_z.median()))
    #print(df.ext_u.std(), (df.ext_g.std()), (df.ext_r.std()), (df.ext_i.std()), (df.ext_z.std()))
    #print(df[df.psf_r < 0])

    '''
    # Remove absolute magnitude dependence, only use differences between bands (i.e. colours)
    df['psf_r_corr_u'] = df.psf_r_corr - df.psf_u_corr
    df['psf_r_corr_g'] = df.psf_r_corr - df.psf_g_corr
    df['psf_r_corr_i'] = df.psf_r_corr - df.psf_i_corr
    df['psf_r_corr_z'] = df.psf_r_corr - df.psf_z_corr
    df['psf_r_corr_w1'] = df.psf_r_corr - df.w1
    df['psf_r_corr_w2'] = df.psf_r_corr - df.w2
    df['psf_r_corr_w3'] = df.psf_r_corr - df.w3
    df['psf_r_corr_w4'] = df.psf_r_corr - df.w4
    '''
    #df['gradient'] = (df.psf_u - df.psf_z)/2

    #For debugging purposes, can limit size of df to <10% of the 2.5 million.
    #df=df[0::10]

    # Add new column to df, with features transformed into 1D
    transform_features(df, feature_columns, n_components=1)

    return df









#-------------------------------------------------
# Main code
#-------------------------------------------------






if __name__ == "__main__": #so you can import this code and run by hand if desired

    # define feature columns used
    # psf magnitudes
    psf = ['psf_u', 'psf_g', 'psf_r', 'psf_i', 'psf_z']
    # cmodel magnitudes
    cmod = ['cmod_u', 'cmod_g', 'cmod_r', 'cmod_i', 'cmod_z']
    # psf magnitudes corrected for extinction
    psf_ext = ['psf_u_corr', 'psf_g_corr', 'psf_r_corr', 'psf_i_corr', 'psf_z_corr']
    # cmodel magnitudes corrected for extinction
    cmod_ext = ['cmod_u_corr', 'cmod_g_corr', 'cmod_r_corr', 'cmod_i_corr', 'cmod_z_corr']
    # WISE magnitudes
    wise = ['w1' ,'w2', 'w3', 'w4']
    # All high S/N resolved bands
    #resolved_highSN = ['resolvedg','resolvedr', 'resolvedi']
    # errors in r
    errors = ['psferr_r', 'cmoderr_r']
    # Magnitude independent colours
    sdss_colours = ['psf_r_corr_u','psf_r_corr_g','psf_r_corr_i','psf_r_corr_z']
    wise_colours = ['psf_r_corr_w1','psf_r_corr_w2','psf_r_corr_w3','psf_r_corr_w4']

    # Select columns to be used as features (typical combinations tested, commented in/out)
    feature_columns = psf + wise + ['resolvedr']
    #feature_columns = sdss_colours + wise_colours + ['resolvedr']
    #feature_columns = psf_ext + wise
    #feature_columns = psf_ext
    #feature_columns = psf_ext + ['resovled_r']
    #feature_columns = wise


    #Input data - comment out after first run to speed up
    '''
    file = 'SDSS_spec_xmwise_all.csv'
    df = load_and_clean_data(file, feature_columns)
    save_obj(df, 'df_cleaned')
    '''

    df = load_obj('df_cleaned')
    print('features used are:')
    print(df[feature_columns].columns)

    #-------------------------------------------------
    # comment out this section once you are satisfied with hyper-parms
    '''
    # test cross-val on hyperparms?
    crossvalidation_hyperparms()
    # Initial test on accuracy vs percent of data trained/numer of trees (this test can take 30 minutes to complete):

    # try with and without class imbalance fix:
    df_g = df[df['class']=='GALAXY'][0::5]
    df_q = df[df['class']=='QSO']
    df_s = df[df['class']=='STAR']
    df = pd.concat([df_g, df_q, df_s])
    print(df['class'].value_counts())
    #-------------------------------------------------
    # Get f1score as function of training range for figure 2 in paper. This takes ~30 mins.
    train_vs_f1score(df, sampleG=True)
    train_vs_f1score(df, sampleG=False)
    # Results are plotted in SDSS_ML_analysis.py, since plotting is much quicker.
    #exit()
    #-------------------------------------------------
    '''

    # Fix machine learning variables used for the rest of the work:
    train_percent = 0.5
    n_jobs=-1 # use max cpus available
    n_estimators = 200 # number of trees in random forest. ~ at least no_feat^2. Set this to 50 if you want quick but decent results for testing/debugging. Use 200 for paper-worthy results (small accuracy increase but takes annoyingly longer if you're testing/debugging). Algorithmic complexity of a Random forest scales linearly with n_estimators.

    # test to fix class imbalance?
    #df_g = df[df['class']=='GALAXY'][0::5]
    #df_q = df[df['class']=='QSO']
    #df_s = df[df['class']=='STAR']
    #df = pd.concat([df_g, df_q, df_s])

    print(df['class'].value_counts())

    # fit random forest. modify mag_split and mag_lim for tests on magnitude limited training
    data_prep_dict_all = prepare_data(df, feature_columns, train_percent=train_percent, ttsplit=True, mag_split=False, mag_lim=18, ttbelow=True)

    pipeline = RF_fit(data_prep_dict_all, n_estimators, n_jobs=-1)
    # apply to test dataset
    classes_pred_all = RF_classify(pipeline, data_prep_dict_all, n_jobs=-1, ttsplit=True, proba=False)
    metrics(data_prep_dict_all, classes_pred_all)
    print('-'*30)

    # get probabilities for the classifications:
    classes_pred_all_proba = RF_classify(pipeline, data_prep_dict_all, n_jobs=-1, ttsplit=True, proba=True)

    # TRAINING AND VALIDATING NOW COMPLETE
    # Save data and models to disk, they are evaulated in: SDSS_ML_analysis.py
    save_obj(pipeline, 'rf_pipeline') # save pipeline to disk for classifying new sources later on:
    #save_obj(df,'df')
    save_obj(data_prep_dict_all, 'data_prep_dict_all')
    save_obj(classes_pred_all, 'classes_pred_all')
    save_obj(classes_pred_all_proba,'classes_pred_all_proba')
    #save_obj(data_prep_dict_boss, 'data_prep_dict_boss')
    #save_obj(data_prep_dict_sdss, 'data_prep_dict_sdss')
    #save_obj(classes_pred_boss, 'classes_pred_boss')
    #save_obj(classes_pred_sdss, 'classes_pred_sdss')

    # append additional derrived quantities to the df
    df_predclass = pd.DataFrame(classes_pred_all, index=data_prep_dict_all['features_test'].index, columns=['class_pred'])
    # Append probabilities to the original df for test data:
    df = df.join(df_predclass, how='left')
    # Get probabilities from the RF classifier:
    df_proba = pd.DataFrame(classes_pred_all_proba, index=data_prep_dict_all['features_test'].index, columns=['prob_g', 'prob_q', 'prob_s'])
    # Append probabilities to the original df for test data:
    df = df.join(df_proba, how='left')
    df['prob_best'] = df[['prob_g', 'prob_q', 'prob_s']].max(axis=1)
    #save_obj(df, 'df_classprobs') # renamed after adding file to zenodo
    save_obj(df, 'df_spec_classprobs')






    # end
