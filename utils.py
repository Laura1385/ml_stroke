import pandas as pd
import numpy as np
import itertools

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

import missingno as msno

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


def bar_graphics (df, column, color, title, rotation=None, ax=None): #(df_stroke, 'gender', 'blue', 'Gender', ax1)\n",
        if ax is None:
            _, ax = plt.subplots()
        count_v = df[column].value_counts()
        ax.bar(count_v.index, count_v.values, color= color)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticklabels(count_v.index, rotation=rotation)
        if rotation:
            ax.set_xticklabels(count_v.index, rotation=rotation)
        else:
            ax.set_xticklabels(count_v.index)
        if column == 'stroke':
            ax.set_xticks([0, 1])
            
            
            