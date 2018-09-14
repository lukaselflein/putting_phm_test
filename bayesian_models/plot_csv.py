import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Find all relevant files
trace_names = []
exclude = ['__py', 'checkpoints', 'git']
for root, dirs, files in os.walk("./csv", topdown=False):
    for fname in files:
        if any(x in fname for x in exclude):
            continue
        else:
            trace_names.append(fname[:-4])
trace_names.sort()

plt.style.use(['seaborn-poster'])
oaksford_parameters = {'p_a': 0.7014, 'p_i': 0.3111, 'p_e': 0.1878, 'p_o': 0.1804, 'p_ent': 0.1076, 'p_err': 0.0122}
hattori_parameters = {'p_a': 0.726, 'p_i': 0.394, 'p_e': 0.202, 'p_o': 0.266, 'p_ent': 0.070, 'p_err': 0.014}

traces, models = dict(), dict()
for trace_name in trace_names:
    in_file = './csv/' + trace_name + '.csv'
    trace = pd.read_csv(in_file)
    # Burn-in and thinning
    trace = trace.iloc[500::5, :]
    traces[trace_name] = trace
print(traces.keys())

for var_name in trace.columns:
    fig = plt.figure(figsize=(18, 16), dpi=50)
    n_pots = 100
    plt.title(var_name)

    for trace_name in trace_names:
        trace = traces[trace_name][var_name]
        hist = plt.hist(trace, n_pots,
                        histtype='step', normed=True, label=trace_name)

        export_data = []
        for i in range(0, len(hist[0])):
            export_data.append([hist[1][i], hist[0][i]])
        export_data = np.array(export_data)
        np.savetxt('hist_data/' + var_name + '_' + trace_name + '.csv',
                   export_data)
        hist_max = hist[0].max()

    if 'nvc' not in var_name:
        plt.plot([oaksford_parameters[var_name]] * 2, [0, hist_max], label='oaksford')
        plt.plot([hattori_parameters[var_name]] * 2, [0, hist_max], label='hattori')

    plt.legend(loc='best')
    plt.savefig('pics/' + var_name + '.png', bbox_inches='tight')

print('Done.')
