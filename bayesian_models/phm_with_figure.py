import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import pickle
import argparse
import sys


def parse_args(args):
    '''
    >>> parse_args(['-f', 'test.txt'])
    Namespace(f='test.txt', g='-1', s=None)
    '''

    parser = argparse.ArgumentParser(description="MCMC sampling of PHM.")
    parser.add_argument('-f', type=str,  metavar='filename',
                        default='/home/lee/cogsci/phm/oaksford_paper_data.csv',
                        help='Path to dataset for fitting')

    parser.add_argument('-n', type=int, metavar='n',
                        default=10, help='number of participants')

    return(parser.parse_args(args))


def get_posterior(data, n=100, draws=2000, n_init=200000, progressbar=True,
                  *args, **kwargs):
    with pm.Model() as model:

        # Define Priors
        p_err = pm.Uniform('p_err', 0, 0.1)  # Upper limit due to normalization
        p_ent = pm.Uniform('p_ent', 0, 1 - 6 * p_err)
        p_a = pm.Uniform('p_a', 0, 1 - 6 * p_err - p_ent)
        p_e = pm.Uniform('p_e', 0, 1 - 6 * p_err - p_ent)
        p_o = pm.Uniform('p_o', 0, 1 - 6 * p_err - p_ent)
        p_i = pm.Uniform('p_i', 0, 1 - 6 * p_err - p_ent)
        nvc_a = pm.Deterministic('nvc_a', 1 - p_a - 6 * p_err - p_ent)
        nvc_i = pm.Deterministic('nvc_i', 1 - p_i - 6 * p_err - p_ent)
        nvc_e = pm.Deterministic('nvc_e', 1 - p_e - 6 * p_err - p_ent)
        nvc_o = pm.Deterministic('nvc_o', 1 - p_o - 6 * p_err - p_ent)

        # Model specification: define all possible moods
        # syll tt-syntax a     i       e      o     NVC
        aa = [p_a,   p_ent, p_err, p_err, nvc_a]
        ai = [p_err, p_a,   p_err, p_ent, nvc_a]
        ia = ai
        ae = [p_err, p_err, p_a,   p_ent, nvc_a]
        ea = ae
        ao = [p_err, p_ent, p_err, p_a,   nvc_a]
        oa = ao
        ii = [p_err, p_i,   p_err, p_ent, nvc_i]
        ie = [p_err, p_err, p_i,   p_ent, nvc_i]
        ei = ie
        io = [p_err, p_ent, p_err, p_i, nvc_i]
        oi = io
        ee = [p_err, p_err, p_e,   p_ent, nvc_e]
        eo = [p_err, p_ent, p_err, p_e, nvc_e]
        oe = eo
        oo = [p_err, p_ent, p_err, p_o, nvc_o]

        # Define the relationship between moods and syllogisms
        moods = [aa, ai, ae, ao, ia, ii, ie, io, ea, ei, ee, eo, oa, oi, oe, oo]
        syllogs = []
        for m in moods:
            # Figure 1
            line = m[0:4] + [p_err] * 4 + [m[-1]]
            syllogs += [line]
            # Figure 2
            line = [p_err] * 4 + m[0:4] + [m[-1]]
            syllogs += [line]

            line = []
            for para in m[0:4]:
                if para == p_err:
                    line += [p_err]
                else:
                    line += [para / 2]
            # Paste this two times
            line *= 2
            # Add NVC
            line += [m[-1]]

            syllogs += [line] * 2

        model_matrix = tt.stack(syllogs)

        # Define likelihood
        pm.Multinomial(name='rates', n=n, p=model_matrix, observed=data)
        map_estimate = pm.find_MAP(model=model)

        trace = pm.sample(draws=draws, njobs=1, start=map_estimate,
                          n_init=n_init, progressbar=progressbar)

        print('Model logp = ', model.logp(map_estimate))
        return model, trace


def get_data(file_name, nr_part):
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

    return(data)


if __name__ == '__main__':
    # Read command line arguments
    args = parse_args(sys.argv[1:])

    # Parse and preprocess data
    data = get_data(file_name=args.f, nr_part=args.n)

    out_name = 'test'
    trace, model = get_posterior(data=data, n=args.n, draws=2000, n_init=20000)
    # Save thinned and burned trace
    # trace = trace[500, 5]
    with open('./' + out_name + '.pickle', 'wb') as pickle_file:
        pickle.dump({'model': model, 'trace': trace}, pickle_file,
                    pickle.HIGHEST_PROTOCOL)
