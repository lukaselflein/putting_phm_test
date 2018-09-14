#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2018 Cognitive Computation Lab
University of Freiburg
Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import numpy as np
import scipy.optimize
import pandas as pd


def phm(p_a, p_i, p_e, p_o, p_ent, p_err):
    '''
    Returns the predictions of the PHM model as 64x5 array given the model parameters.
    '''

    # Model specification: define all possible moods following CO99
    #      a     i       e     o
    aa = [p_a, p_ent, p_err, p_err]
    ai = [p_err, p_a, p_err, p_ent]
    ae = [p_err, p_err, p_a, p_ent]
    ao = [p_err, p_ent, p_err, p_a]
    ia = ai
    ii = [p_err, p_i, p_err, p_ent]
    ie = [p_err, p_err, p_i, p_ent]
    io = [p_err, p_ent, p_err, p_i]
    ea = ae
    ei = ie
    ee = [p_err, p_err, p_e, p_ent]
    eo = [p_err, p_ent, p_err, p_e]
    oa = ao
    oi = io
    oe = eo
    oo = [p_err, p_ent, p_err, p_o]

    # Define the relationship between moods and syllogisms
    # Khemlani-ordering:
    # moods = [aa, ai, ae, ao, ia, ii, ie, io, ea, ei, ee, eo, oa, oi, oe, oo]

    # Ordering of moods like in CO99 data
    moods = [aa, ai, ia, ae, ea, ao, oa, ii, ie, ei, io, oi, ee, eo, oe, oo]
    syllogs = []
    for m in moods:
        row = []
        row += m
        # Introduce a NVC paramater to normalize the rows
        row += [1 - sum(row)]
        # Only 5 answers possible, thus all figures have same prediction in PHM
        syllogs += [row] * 4

    # Now we have a list of 64 lists of 5 numbers each. Convert this to a numpy array
    predictions = np.array(syllogs)
    return predictions


def rmse(parameters, model, target_data):
    '''Root Mean Squared Error on all of the data'''
    predictions = model(*parameters)
    error = np.sqrt(np.mean((predictions - target_data)**2))
    return(error)


def rmse_without_nvc(parameters, model, target_data):
    '''Root Mean Squared Error neglecting the NVC response (column 4)'''
    predictions = model(*parameters)
    error = np.sqrt(np.mean((predictions[:, 0:4] - target_data[:, 0:4])**2))
    return(error)


class PHM():
    """
    Wrapper for the phm function.
    Supports fitting and predicting model responses.
    """

    def fit(self, dataset, method):
        """
        Fits, saves and returns internal parameters.
        """
        self.dataset = dataset
        init_para = [0.25] * 6
        bounds = [(0., 1.)] * 6
        res = scipy.optimize.minimize(method, init_para, method='L-BFGS-B',
                                      bounds=bounds, options={'disp': False},
                                      args=(phm, dataset))
        self.parameters = res.x
        return self.parameters

    def predict(self, problem):
        """
        Returns the model's prediction on all syllogims as 64x5 matrix.
        """
        predictions = phm(*self.parameters)
        return predictions


if __name__ == '__main__':
    # Read data
    data_path = '../data/co99/co99_paper_data.csv'
    df = pd.read_csv(data_path)
    # Rename columns
    df.columns = ('syllog', 'a', 'mod_a', 'i', 'mod_i', 'e', 'mod_e', 'o', 'mod_o', 'nvc')
    # Remove syllogism labels AA1...
    df.pop('syllog')
    # Convert string to float
    df.astype(float)
    # Convert percentages [0, 100] to [0, 1]
    df = df.multiply(0.01)

    # Select the CO99 'Model' column
    mod_data = df[['mod_a', 'mod_i', 'mod_e', 'mod_o']].copy()
    # Use the NVC column as normalization parameter
    mod_data['nvc'] = 1 - mod_data.sum(axis=1)
    # Convert to numpy
    mod_data = mod_data.as_matrix()

    # Select the 'Data' column
    exp_data = df[['a', 'i', 'e', 'o']].copy()
    # Normalize experimental data via NVC
    # ATTENTION: CO99 data is not normalized!
    # Misc answers will be treated as 'NVC' answers when normalizing this way!
    exp_data['nvc'] = 1 - exp_data.sum(axis=1)
    # convert to numpy
    exp_data = exp_data.as_matrix()

    datasets = {'exp': exp_data, 'mod': mod_data}

    print('Optimizing on RMSE without NVC, similar to CO99')
    model = PHM()
    model.fit(exp_data, method=rmse_without_nvc)
    print('p_A,\t p_I,\t p_E,\t p_O,\t p_ent,\t p_err')
    print(('{:1.2f},\t ' * 6).format(*model.parameters))
    prediction = model.predict(problem=None)
    no_nvc_val = np.sqrt(np.mean((prediction[:, 0:4] - exp_data[:, 0:4])**2))
    print('RMSE-NVC = {:1.3f}'.format(no_nvc_val))
    print('RMSE on full data = {:1.3f} '.format(np.sqrt(np.mean((prediction - exp_data)**2))))
    print()

    print('Optimizing on full data')
    model.fit(exp_data, method=rmse)
    print('p_A,\t p_I,\t p_E,\t p_O,\t p_ent,\t p_err')
    print(('{:1.2f},\t ' * 6).format(*model.parameters))
    prediction = model.predict(problem=None)
    no_nvc_val = np.sqrt(np.mean((prediction[:, 0:4] - exp_data[:, 0:4])**2))
    print('RMSE-NVC = {:1.3f}'.format(no_nvc_val))
    print('RMSE on full data = {:1.3f} '.format(np.sqrt(np.mean((prediction - exp_data)**2))))
