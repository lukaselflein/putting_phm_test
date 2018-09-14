import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import pickle
import argparse
import sys


def parse_args(args):
    '''
    Parses command line arguments:
    -f, a different data file
    -n, the number of participants in this data file
    '''

    parser = argparse.ArgumentParser(description="MCMC sampling of PHM.")
    parser.add_argument('-f', type=str,  metavar='filename',
                        default='/home/lee/cogsci/phm/oaksford_paper_data.csv',
                        help='Path to dataset for fitting')

    parser.add_argument('-n', type=int, metavar='nr_part',
                        default=10, help='number of participants')

    return(parser.parse_args(args))


def get_posterior(data, n=1, draws=1000, n_init=10000, *args, **kwargs):
    '''
    Returns a pymc3 MCMC trace for the PHM model.
    data is a 64x5 matrix of syllogism answer frequencies.
    n is the number of participants in the data.
    draws is the number of MCMC samples.
    n_init is the number of samples for the ADVI initialization.
    '''
    with pm.Model() as model:

        # Define Priors
        p_err = pm.Uniform('p_err', 0, 0.3)  # Upper limit due to normalization
        p_ent = pm.Uniform('p_ent', 0, 1 - 2 * p_err)  # Upper limit depends on error prob
        p_a = pm.Uniform('p_a', 0, 1 - 2 * p_err - p_ent)
        p_e = pm.Uniform('p_e', 0, 1 - 2 * p_err - p_ent)
        p_o = pm.Uniform('p_o', 0, 1 - 2 * p_err - p_ent)
        p_i = pm.Uniform('p_i', 0, 1 - 2 * p_err - p_ent)
        nvc_a = pm.Deterministic('nvc_a', 1 - p_a - 2 * p_err - p_ent)
        nvc_i = pm.Deterministic('nvc_i', 1 - p_i - 2 * p_err - p_ent)
        nvc_e = pm.Deterministic('nvc_e', 1 - p_e - 2 * p_err - p_ent)
        nvc_o = pm.Deterministic('nvc_o', 1 - p_o - 2 * p_err - p_ent)

        # Model specification: define all possible moods
        # syll tt-syntax a     i       e      o     NVC
        aa = tt.stack(p_a,   p_ent, p_err, p_err, nvc_a)
        ai = tt.stack(p_err, p_a,   p_err, p_ent, nvc_a)
        ia = ai
        ae = tt.stack(p_err, p_err, p_a,   p_ent, nvc_a)
        ea = ae
        ao = tt.stack(p_err, p_ent, p_err, p_a,   nvc_a)
        oa = ao
        ii = tt.stack(p_err, p_i,   p_err, p_ent, nvc_i)
        ie = tt.stack(p_err, p_err, p_i,   p_ent, nvc_i)
        ei = ie
        io = tt.stack(p_err, p_ent, p_err, p_i, nvc_i)
        oi = io
        ee = tt.stack(p_err, p_err, p_e,   p_ent, nvc_e)
        eo = tt.stack(p_err, p_ent, p_err, p_e, nvc_e)
        oe = eo
        oo = tt.stack(p_err, p_ent, p_err, p_o, nvc_o)

        # Define the relationship between moods and syllogisms
        moods = [aa, ai, ia, ae, ea, ao, oa, ii, ie, ei, io, oi, ee, eo, oe, oo]
        syllogs = []
        for m in moods:
            # Add each mood four times (4 figures)
            syllogs += [m] * 4

        # Turn list into a theano tensor
        model_matrix = tt.stack(syllogs)
        print('Model specified')

        # Define likelihood
        # n = np.array([data.sum(axis=1)] * 5).T
        pm.Multinomial(name='rates', n=n, p=model_matrix, observed=data)
        print('Likelihood specified')

        trace = pm.sample(draws=draws, njobs=1, n_init=n_init)

        map_estimate = pm.find_MAP(model=model)
        print('Model logp = ', model.logp(map_estimate))
        return model, trace


def get_data(file_name, nr_part):
    '''
    Parses and converts experimental data.
    Returns 64x5 array of answer numbers (int).
    '''
    df = pd.read_csv(file_name)
    # Rename columns
    df.columns = ('syllog', 'a', 'i', 'e', 'o', 'nvc')
    df.pop('syllog')
    df.astype(float)

    # Number of participants
    df = df.multiply(nr_part)
    df = df.round().astype(int)

    # Select the experimental data only !! NOT NORMALIZED, sum may be greater or less!!
    # exp_data = df[['a', 'i', 'e', 'o', 'nvc']]

    # Normalize experimental data via NVC
    data = df[['a', 'i', 'e', 'o']].copy()
    data['nvc'] = nr_part - data.sum(axis=1)

    return data


if __name__ == '__main__':
    # Read command line arguments
    args = parse_args(sys.argv[1:])

    # Parse and preprocess data
    data = get_data(file_name=args.f, nr_part=args.nr_part)

    out_name = 'test'
    trace, model = get_posterior(data=data, n=100, draws=2000)
    with open('./' + out_name + '.pickle', 'wb') as pickle_file:
        pickle.dump({'model': model, 'trace': trace}, pickle_file,
                    pickle.HIGHEST_PROTOCOL)
