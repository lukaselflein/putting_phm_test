#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2018 Cognitive Computation Lab
University of Freiburg
Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import pandas as pd
import numpy as np
import phm


def is_aieo(raw):
    '''
    Check if Most or Few quantifier is absent.
    '''
    types = raw['Type'][5]
    for i in range(0, len(types)):
        if 'F' in types[i] or 'M' in types[i]:
            return False
    return True


def get_hattori_data(folder_path='../data/ha16/'):
    '''
    Parse datasets collected by Hattori (2016).
    Reject data with non-normalized rows or Most/Few quantifiers or not in 64x5 format.
    Returns 64x5 matrix of aggregated data.
    '''
    file_names = ['BBJ95-adu.csv', 'CO99-1.csv', 'CO99-2.csv', 'CO99.csv',
                  'D78-1.csv', 'D78-2.csv', 'GS81-1.csv', 'H17-1.csv',
                  'H17-2.csv', 'JB84-3.csv', 'JS78-2a.csv', 'JS78-2b.csv',
                  'RNG01.csv']
    data = {}
    for name in file_names:
        # Read raw data via pandas
        raw = pd.read_csv(folder_path + name)
        # Check for Most/Few quantifiers and 64x5 format.
        if is_aieo(raw) and raw.shape == (64, 6):
            raw.columns = ('Type', 'a', 'i', 'e', 'o', 'nvc')
            # Percentages instead of [0,1]
            raw = raw[['a', 'i', 'e', 'o', 'nvc']]
            # Check if all rows are normalized, i.e. if there are no 'MISC' responses
            if any(raw.sum(axis=1) < 0.99):
                print('Skipping {}, rows not normalized to 1'.format(name))
                continue
            # Convert to numpy array
            data[name[:-4]] = raw.as_matrix()
    return data


def get_rg_data(file_name='../data/rg16/rg16.csv'):
    '''
    Parse and aggregate rg16 dataset.
    Return 64x5 array.
    '''
    syllogisms = generate_syllogisms()
    # File seperator is ';' instead of ','
    df = pd.read_csv(file_name, sep=';')
    # Use only syllogisms, not the 4 sham questions
    mask = [x in syllogisms for x in df['syllog']]
    df = df.loc[mask]  # [['syllog', 'givenanswer']]
    # Sort according to  syllogism ordering
    # Create the dictionary that defines the order for sorting
    sorter_index = dict(zip(syllogisms, range(len(syllogisms))))
    # Generate a rank column that will be used to sort
    # the dataframe numerically
    df['Tm_Rank'] = df['syllog'].map(sorter_index)
    df = df.sort_values(by='Tm_Rank')
    # Get participant codes
    participants = list(df.code.unique())
    agg_answers = np.array([[0] * 5] * 64)
    for part in participants:
        # Get participant data
        part_data = df.loc[df['code'] == part]
        for index in range(0, 64):
            quantifier = part_data.iloc[index]['ConclQ']
            if quantifier == 'A':
                chosen_answer = 0
            elif quantifier == 'I':
                chosen_answer = 1
            elif quantifier == 'E':
                chosen_answer = 2
            elif quantifier == 'O':
                chosen_answer = 3
            elif quantifier == 'NVC':
                chosen_answer = 4
            agg_answers[index, chosen_answer] += 1

    return agg_answers / len(participants)


def generate_syllogisms():
    '''Return all 64 syllogism labels in Khemlani (2012) ordering'''
    quantifiers = ('A', 'I', 'E', 'O')
    syllog_labels = []
    for first in quantifiers:
        for second in quantifiers:
            for figure in ('1', '2', '3', '4'):
                syllog_labels += [first + second + figure]
    return syllog_labels


def rmse(parameters, model, target_data):
    '''Root Mean Squared Error'''
    predictions = model(*parameters)
    error = np.sqrt(np.mean((predictions - target_data)**2))
    return error


def rmse_without_nvc(parameters, model, target_data):
    '''Root Mean Squared Error ignoring NVC'''
    predictions = model(*parameters)
    error = np.sqrt(np.mean((predictions[:, 0:4] - target_data[:, 0:4])**2))
    return error


def rmse_hattori(parameters, model, target_data):
    '''Calculates Hattori's modified Root Mean Squared Error on model'''
    predictions = model(*parameters)
    error = rmse_val_hattori(predictions, target_data)
    return error


def rmse_val_hattori(predictions, target_data):
    ''' Returns value of Hattori's modified Root Mean Squared Error'''
    error = 0.
    for i in range(0, 64):
        error += np.sqrt(np.mean((predictions[i, :] - target_data[i, :])**2))
    error /= 64
    return error


def rmse_val(predictions, target_data):
    '''Returns value of Root Mean Squared Error'''
    return np.sqrt(np.mean((predictions - target_data)**2))


if __name__ == '__main__':
    # Parse and preprocess RG data
    rg_data = get_rg_data()
    rg_df = pd.DataFrame(rg_data)

    # Get hattori datasets as dict
    datasets = get_hattori_data()

    # Add aggregated 139 data to data sets
    datasets['rg16'] = rg_data

    # Define the row indices for the .csv file
    indices = sorted(list(datasets.keys()))
    # Define the column names for the .csv file
    columns = ['rmse', 'rmse_hattori', 'p_a', 'p_i', 'p_e', 'p_o', 'p_ent', 'p_err']
    # Create a pandas Data Frame for csv export
    rmse_df = pd.DataFrame(columns=columns, index=indices)
    # Copy it
    dfs = {'rmse': rmse_df, 'rmse_hattori': rmse_df.copy()}

    # Calculate the fit errors on all datasets:
    for name in sorted(list(datasets.keys())):
        # once with standard RMSE and once with the method of Hattori (2016)
        for method in rmse, rmse_hattori:
            data = datasets[name]
            model = phm.PHM()
            model.fit(data, method=method)
            prediction = model.predict(problem=None)
            df = dfs[method.__name__]
            df.loc[name][:] = pd.Series([rmse_val(prediction, data),
                                         rmse_val_hattori(prediction, data),
                                         *model.parameters])
    # Save everything as .csv
    for name in dfs.keys():
        df = dfs[name]
        df.to_csv(name + '_minimized_parameters.csv', sep=',')
