import pickle
import os
import pandas as pd

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

# Get parameter traces
for index in range(len(trace_names)):
    trace_name = trace_names[index]
    print('loading {} of {}'.format(index + 1, len(trace_names)), end='\r')
    with open('./traces/' + trace_name + '.pickle', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        # Select trace data
        trace = data['trace']
        var_dic = {}
        # Remove interval variables
        for var_name in trace.varnames:
            if 'interval' in var_name:
                continue
            var_dic[var_name] = trace[var_name]

        # Convert to pandas DataFrame
        df = pd.DataFrame.from_dict(var_dic)
        out_name = './csv/' + trace_name + '.csv'
        df.to_csv(out_name, sep=',', header=True, index=False, decimal='.')

        trace = None
        data = None

print('\nDone.')
