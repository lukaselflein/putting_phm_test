import pickle
import os
import matplotlib.pyplot as plt


oaksford_parameters = {'p_a': 0.7014, 'p_i': 0.3111, 'p_e': 0.1878, 'p_o': 0.1804, 'p_ent': 0.1076, 'p_err': 0.0122}
hattori_parameters = {'p_a': 0.726, 'p_i': 0.394, 'p_e': 0.202, 'p_o': 0.266, 'p_ent': 0.070, 'p_err': 0.014}

print('Import done.')
# Find all relevant files
trace_names = []
exclude = ['__py', 'checkpoints', 'git']
for root, dirs, files in os.walk("./traces", topdown=False):
    for fname in files:
        if any(x in fname for x in exclude):
            continue
        else:
            trace_names.append(fname[:-7])
trace_names.sort()
print('searching done.')

# Parameter means dictionary
para_means = {}

# Get parameter names
with open('./traces/' + trace_names[-1] + '.pickle', 'rb') as pickle_file:
    data = pickle.load(pickle_file)
    trace = data['trace']
    for var_name in trace.varnames:
        if 'interval' in var_name:
            continue

        para_means[var_name] = {}

# Get parameter means from traces
for index in range(len(trace_names)):
    trace_name = trace_names[index]
    print('loading {} of {}'.format(index, len(trace_names)))
    with open('./traces/' + trace_name + '.pickle', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        # Select trace data
        trace = data['trace']
        for var_name in trace.varnames:
            if 'interval' in var_name:
                continue
            para_means[var_name][trace_name] = trace[var_name].mean()

        trace = None

for name in para_means['p_i']:
    print(name, para_means['p_i'][name])
exit()

for var_name in para_means.keys():
    fig = plt.figure(figsize=(18, 16), dpi=50)
    plt.title(var_name)

    hist = plt.hist(para_means[var_name].values(), 100,
                    histtype='step', normed=False,
                    label='rg16')

    hist_max = hist[0].max()
    if 'nvc' not in var_name:
        plt.plot([oaksford_parameters[var_name]] * 2, [0, hist_max], label='oaksford')
        plt.plot([hattori_parameters[var_name]] * 2, [0, hist_max], label='hattori')

    plt.legend(loc='best')
    plt.savefig('para_pics/' + var_name + '.png', bbox_inches='tight')

print('Done.')
