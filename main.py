#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 19:03:25 2022

@author: ymonjid
"""

import pandas as pd
import numpy as np
from Data_cleaning import data_cleaning

# 1) Reading the Data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# 2) Cleaning the Data
df_train_clean = data_cleaning(df_train)