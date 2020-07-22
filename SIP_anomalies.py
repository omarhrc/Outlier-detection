#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:56:14 2020

@author: omarhrc
"""
from sklearn.preprocessing import StandardScaler, Normalizer
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyod.models.mo_gaal import MO_GAAL
#from pyod.utils.data import evaluate_print
#from pyod.utils.example import visualize

########################################################################################################
#
# Input file
#
########################################################################################################
IN_PATH = r'/home/omarhrc/Documents/AI/Python scripts/Anomaly Detection'
IN_FILE = 'sip_vectors.csv'

########################################################################################################
#
# Output file
#
########################################################################################################
OUT_PATH = r'/home/omarhrc/Documents/AI/Python scripts/Anomaly Detection'
OUT_FILE = 'outliers.xls'

########################################################################################################
#
# Other constants
#
########################################################################################################

TIME_FIELDS = ['Start timestamp', 'Setup Time', 'Answer Time', 'Call Duration', 'ST Valid', 'AT Valid']

df_all = pd.read_csv(os.path.join(IN_PATH, IN_FILE), delimiter = '\t')
df_all['Start timestamp'] = pd.to_datetime(df_all['Start timestamp'], unit='s')
df = df_all.loc[:,TIME_FIELDS[1]:]
X = df.to_numpy()
scaler = StandardScaler()

# Representation 1 & 4
#X_scaled = scaler.fit_transform(X)

# Representation 2
# times & TFIDF: Scale times only separately
#X_times_scaled = scaler.fit_transform(X[:,:len(TIME_FIELDS)])
#X_scaled = np.hstack([X_times_scaled, X[:,len(TIME_FIELDS):]])

# Representation 3
# times & TFIDF: Scale all
# X_scaled = scaler.fit_transform(X)

# Representation 5
# times & TFIDF: Scale times only and TFIDF features separately. TFIDF vectors are made L2 norm = 1
X_times_scaled = scaler.fit_transform(X[:, :len(TIME_FIELDS) - 1])
scaler_tfidf = Normalizer()
X_tfidf = scaler_tfidf.fit_transform(X[:, len(TIME_FIELDS) - 1:])
X_scaled = np.hstack([X_times_scaled, X_tfidf])


clf = MO_GAAL(contamination = 0.05)
clf.fit(X_scaled)

df_all['scores'] = clf.decision_scores_
df_all['labels'] = clf.labels_
df_out = df_all.where(df_all['labels'] == 1).dropna()
df_out = df_out.loc[:, (df_out != 0).any(axis=0)]
df_out.to_excel(os.path.join(OUT_PATH, OUT_FILE))
