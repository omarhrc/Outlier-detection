#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:56:14 2020

@author: omarhrc
"""
import pyshark
from io import StringIO
import sys
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os.path
import platform
import datetime
import numpy as np
import scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import numply as numpyiiii


########################################################################################################
#
# Config files
#
########################################################################################################
FIELDS_PATH = r'/home/omarhrc/Documents/AI/Python scripts/Anomaly Detection'
FIELDS_FILE = 'sip_fields_ chosen.txt'
FIELDS_LIM_FILE = 'sip_fields_limited.txt'

########################################################################################################
#
# Input file
#
########################################################################################################
PCAP_PATH = r'/home/omarhrc/Documents/AI/Python scripts/Anomaly Detection'
PCAP_FILE = 'VM - International calls - 090720.pcapng'
#PCAP_FILE = 'siptest_short.pcapng'


########################################################################################################
#
# Output file
#
########################################################################################################
OUT_PATH = r'/home/omarhrc/Documents/AI/Python scripts/Anomaly Detection'
OUT_FILE = 'sip_vectors.csv'

########################################################################################################
#
# Other constants
#
########################################################################################################

LINE_SEPARATOR = '\r\n'
if platform.system() == 'Linux':
    LINE_SEPARATOR = '\n'

NUM_RANGE_LEN = 5

TIME_FIELDS = ['Start timestamp', 'Setup Time', 'Answer Time', 'Call Duration', 'ST Valid', 'AT Valid', 'Outgoing']
TIME_FIELDS_IDX = {time_field:idx for idx, time_field in enumerate(TIME_FIELDS)}

REF_RANGE = '+447' 

########################################################################################################
#
# Functions
#
########################################################################################################


def get_attr_val(line, strip_line = True, separator = ':', max_lenght = NUM_RANGE_LEN, max_lenght_attr = set()):
        ''' Gets attribute : value pair from a line'''
        # Splits line
        split_line = line.split(':')
        # Checks string
        elements = len(split_line)
        if (elements == 0):
                attr, val = None, None
        elif (elements == 1):
                attr, val = split_line[0], None
        else:
                attr, val = split_line[0], split_line[1]
        # Strips values, if needed
        if strip_line:
                if attr is not None:
                        attr = attr.strip()
                if val is not None:
                        val = val.strip()
        # Limits value lenghts
        if (val is not None) and (len(val) > max_lenght) and (attr in max_lenght_attr):
                val = val[:max_lenght]
        return attr, val

        

########################################################################################################
#
# Main program
#
########################################################################################################
                
# Reads fields
with open(os.path.join(FIELDS_PATH, FIELDS_FILE), 'r') as f:
        field_names = set([field.strip() for field in f.readlines()])

# Reads fileds to be shortened
with open(os.path.join(FIELDS_PATH, FIELDS_LIM_FILE), 'r') as f:
        field_lim_names = set([field.strip() for field in f.readlines()])

# Reads cap files
old_stdout = sys.stdout
cap = pyshark.FileCapture(os.path.join(PCAP_PATH, PCAP_FILE), keep_packets = False)
calls = []
prev_ID = None
call_times = [0]*len(TIME_FIELDS)
r_uri_checked = False
for pkt in cap:
        # Get full printout
        sys.stdout = mystdout = StringIO()
        print(pkt.sip.pretty_print())
        sip_msg = mystdout.getvalue()
        mystdout.close()
        sys.stdout = old_stdout
        # Extracts info from msg
        msg_dict = defaultdict(list)
        for line in sip_msg.split(LINE_SEPARATOR):
                attr, val = get_attr_val(line, max_lenght_attr = field_lim_names)
                if val is None:
                        continue
                if attr in field_names:
                        msg_dict[attr].append(val)
        msg_dict['timestamp'] = pkt.sniff_time
        # Updates incoming/outgoing attribute
        if not r_uri_checked:
            try:
                call_times[TIME_FIELDS_IDX['Start timestamp']] = pkt.sniff_timestamp
                call_times[TIME_FIELDS_IDX['Outgoing']] = 0 if (REF_RANGE in pkt.sip.r_uri_user) else 1
                r_uri_checked = True
            except AttributeError:
                pass
        # Updates call_times
        try:
            status_code = int(pkt.sip.status_code)
            if (call_times[TIME_FIELDS_IDX['ST Valid']] == 0):
                if (100 < status_code < 300) and (pkt.sip.cseq_method == 'INVITE'):
                    # Setup successful. Update validity flag and time
                    call_times[TIME_FIELDS_IDX['ST Valid']] = 1
                    call_times[TIME_FIELDS_IDX['Setup Time']] = (pkt.sniff_time - calls[-1][0]['timestamp']).total_seconds()
            elif (call_times[TIME_FIELDS_IDX['AT Valid']] == 0):
                if (200 <= status_code < 300) and (pkt.sip.cseq_method == 'INVITE'):
                    # Answer successful. Update validity flag and time
                    call_times[TIME_FIELDS_IDX['AT Valid']] = 1
                    call_times[TIME_FIELDS_IDX['Answer Time']] = (pkt.sniff_time - calls[-1][0]['timestamp']).total_seconds()
        except AttributeError:
            pass 
        # Group per SIP Call_ID
        if (prev_ID is None) or (prev_ID != pkt.sip.call_id):
            if (prev_ID is not None):
                call_times[TIME_FIELDS_IDX['Call Duration']] = (calls[-1][-1]['timestamp'] - calls[-1][0]['timestamp']).total_seconds()
                calls[-1].append(call_times)
            # New call
            calls.append([msg_dict])
            prev_ID = pkt.sip.call_id
            call_times = [0]*len(TIME_FIELDS)
            r_uri_checked = False
        else:
                # Same call
                calls[-1].append(msg_dict)
if len(calls):
    call_times[TIME_FIELDS_IDX['Call Duration']] = (calls[-1][-1]['timestamp'] - calls[-1][0]['timestamp']).total_seconds()    
    calls[-1].append(call_times)
cap.close()

####################################################################################################
#
# Representation 1:
# One vector per call:
# - Concatenate all fields from all messages and make document out of it
# - All documents will be input for a Counter vectorizer
# - Vectorizer will transform each document (call) into a vector
#
####################################################################################################
# docs = []
# X_times = np.zeros([len(calls), len(TIME_FIELDS)])
# for idx, call in enumerate(calls):
#         doc = ''
#         for msg in call:
#             if isinstance(msg, defaultdict):
#                 # These are msg fields
#                 for vals in msg.values():        # Gets a list of vals
#                     try:
#                         for val in vals:        # Gets each val
#                                     doc = '\t'.join([doc, val])
#                     except TypeError:
#                         continue
#             else:
#                 # These are timestamps
#                 X_times[idx] = msg
#         docs.append(doc)
# vectorizer = CountVectorizer(lowercase = False, tokenizer=lambda x: x.split('\t'))
# X = vectorizer.fit_transform(docs)
# map_idx = {val:key for key, val in vectorizer.vocabulary_.items()}
# list_idx = [map_idx[i] for i in range(len(map_idx))]
# X = scipy.sparse.hstack([X_times, X])

####################################################################################################
#
# Representation 2 & 3 & 5:
# One vector per call:
# - Concatenate all fields from all messages and make document out of it
# - All documents will be input for a TFIDF vectorizer
# - Vectorizer will transform each document (call) into a vector
#
####################################################################################################
docs = []
X_times = np.zeros([len(calls), len(TIME_FIELDS)])
for idx, call in enumerate(calls):
        doc = ''
        for msg in call:
            if isinstance(msg, defaultdict):
                # These are msg fields
                for vals in msg.values():        # Gets a list of vals
                    try:
                        for val in vals:        # Gets each val
                            if (REF_RANGE not in val):
                                #print (val)
                                doc = '\t'.join([doc, val])
                    except TypeError:
                        continue
            else:
                # These are timestamps
                X_times[idx] = msg
        docs.append(doc)
vectorizer = TfidfVectorizer(lowercase = False, tokenizer=lambda x: x.split('\t'))
X = vectorizer.fit_transform(docs)
map_idx = {val:key for key, val in vectorizer.vocabulary_.items()}
list_idx = [map_idx[i] for i in range(len(map_idx))]
X = scipy.sparse.hstack([X_times, X])

####################################################################################################
#
# Representation 4:
# One vector per call:
# - Concatenate all fields from all messages and make document out of it
# - All documents will be input for a Counter vectorizer with binary features
# - Vectorizer will transform each document (call) into a vector
#
####################################################################################################
##docs = []
##X_times = np.zeros([len(calls), len(TIME_FIELDS)])
##for idx, call in enumerate(calls):
##        doc = ''
##        for msg in call:
##            if isinstance(msg, defaultdict):
##                # These are msg fields
##                for vals in msg.values():        # Gets a list of vals
##                    try:
##                        for val in vals:        # Gets each val
##                                    doc = '\t'.join([doc, val])
##                    except TypeError:
##                        continue
##            else:
##                # These are timestamps
##                X_times[idx] = msg
##        docs.append(doc)
##vectorizer = CountVectorizer(lowercase = False, tokenizer=lambda x: x.split('\t'), binary = True)
##X = vectorizer.fit_transform(docs)
##map_idx = {val:key for key, val in vectorizer.vocabulary_.items()}
##list_idx = [map_idx[i] for i in range(len(map_idx))]
##X = scipy.sparse.hstack([X_times, X])


##########################################
# Get ready for EDA and write to csv file
##########################################
df = pd.DataFrame(X.toarray(), columns = TIME_FIELDS + list_idx)
#df['Start timestamp'] = pd.to_datetime(df['Start timestamp'], unit='s')
df.to_csv(os.path.join(OUT_PATH, OUT_FILE), sep = '\t', index = False)

print('-' * 40)
print(df['Setup Time'].where(df['ST Valid'] == 1).describe())
print('-' * 20)
print(df['Answer Time'].where(df['AT Valid'] == 1).describe())
print('-' * 20)
print(df['Call Duration'].describe())
print('-' * 40)

#########################################
# How to see a particular record
#########################################
df_ST = df['Setup Time'].where(df['ST Valid'] == 1)
df_ST_max = df.where(df['Setup Time'] == df_ST.max()).dropna()
ST_max = df_ST_max[df_ST_max.columns[(df_ST_max.values != 0).any(axis=0)]].T
print (f'Max Setup Time:\n{ST_max}')
print('-' * 40)

fig, axes = plt.subplots(nrows=1,ncols=3)


fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
axes[0].set(title = 'Setup Time', xlabel = 'secs')
axes[1].set(title = 'Answer Time', xlabel = 'secs')
axes[2].set(title = 'Call Duration', xlabel = 'secs')
df['Setup Time'].where(df['ST Valid'] == 1).plot.hist(bins = 10, ax = axes[0], subplots=True)
df['Answer Time'].where(df['AT Valid'] == 1).plot.hist(bins = 10, ax = axes[1], subplots=True)
df['Call Duration'].plot.hist(bins = 10, ax = axes[2], subplots=True)
plt.plot()

#########################################
# Finds outliers
#########################################

# classpyod.models.mo_gaal.MO_GAAL(k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, decay=1e-06, momentum=0.9, contamination=0.1)[source]
# clf = MO_GAAL(contamination = 0.01)
# clf.fit(X)


