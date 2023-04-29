import pandas as pd
import numpy as np
import itertools
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 

from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate, RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import loguniform, beta, uniform
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as IMBPipeline

import warnings
warnings.filterwarnings('ignore')


#Bar's graphic
def bar_graphics(df, column, title, rotation=None, ax=None, y_max=3000):
    if ax is None:
        _, ax = plt.subplots()
    count_v = df[column].value_counts()
    colors = plt.cm.Dark2(np.arange(len(count_v))) #Set3
    ax.bar(count_v.index, count_v.values, color=colors)
    ax.set_ylabel('Count')
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels(count_v.index, rotation=rotation)
    if rotation:
        ax.set_xticklabels(count_v.index, rotation=rotation)
    else:
        ax.set_xticklabels(count_v.index)
    #%above the bars
    for i, v in enumerate(count_v.values):
        ax.text(i, v + 0.03*max(count_v.values), f"{(v/sum(count_v.values))*100:.2f}%", color='black', fontweight='bold', ha='center')
    
    if column == 'stroke':
        ax.set_xticks([0, 1])
    
    if y_max:
        ax.set_ylim([0, y_max])
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    

#Correlation_matrix's graphic
def correlation_matrix(df, title):
    data = df
    corr = data.corr() #os need (numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    #Create the mask to NOT have graphic redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool), k = 1)
    
    #Create correlation palette and heat map
    cmap = sns.color_palette('Spectral_r', as_cmap=True)
    dataplot = sns.heatmap(corr, mask=mask, center=0, annot=True, annot_kws={'size': 8}, fmt='.1f', cmap=cmap, linewidths=.35)
    plt.title(title, fontsize=15)
    plt.show()