import pandas as pd
import numpy as np
import pickle
import phm
import os
import multiprocessing as mp
import time


def is_aieo(raw):
    types = raw['Type'][5]
    for i in range(0, len(types)):
        if 'F' in types[i] or 'M' in types[i]:
            return(False)
    return(True)


def get_hattori_data():
    folder_path = '../data/ha16/'
    file_names = ['BBJ95-adu.csv', 'CO99-1.csv', 'CO99-2.csv', 'CO99.csv',
                  'D78-1.csv', 'D78-2.csv', 'GS81-1.csv', 'H17-1.csv',
                  'H17-2.csv', 'JB84-3.csv', 'JS78-2a.csv', 'JS78-2b.csv',
                  'RNG01.csv']

    # Get number of participants for each experiment
    participant_numbers = pd.read_csv(folder_path + 'participant_numbers.csv')
    # Transform the pandas.DataFrame to a python dictionary
    part_nr = participant_numbers.set_index('Filename').T.to_dict('int')
    part_nr = part_nr[list(part_nr.keys())[0]]

    # Read the experimental data sets
    data = {}
    for name in file_names:
        raw = pd.read_csv(folder_path + name)
        if is_aieo(raw) and raw.shape == (64, 6):
            raw.columns = ('Type', 'a', 'i', 'e', 'o', 'nvc')
            # Percentages instead of [0,1]
            n = part_nr[name]
            raw = raw[['a', 'i', 'e', 'o', 'nvc']] * n
            # Cast the data to int, ie. treat it as number of answers
            # Source of error: the lines do not have the same normalization, i.e.
            # some syllogs have different total numbers of answers.
            # Careful when treating these data in probabilistic models!
            if any(raw.sum(axis=1) != n):
                print('Skipping {}, rows not normalized to {}'.format(name, n))
                continue
            data[name[:-4]] = raw.astype(int)
            print('Addded', name)

        elif raw.shape != (64, 6):
            print('Skipping {}, has shape {}'.format(name, raw.shape))
        else:
            print('Skipping {}, contains M or F'.format(name))
    return data


def get_rg_data(file_name='/home/lee/cogsci/data/rg_2016/syllo64_dataFil_raw.csv'):
    syllogisms = generate_syllogisms()
    # File seperator is ';' instead of ','
    df = pd.read_csv(file_name, sep=';')
    # Use only syllogisms, not the 4 sham questions
    mask = [x in syllogisms for x in df['syllog']]
    df = df.loc[mask]  # [['syllog', 'givenanswer']]
    # Sort according to  syllogism ordering
    # Create the dictionary that defines the order for sorting
    sorterIndex = dict(zip(syllogisms, range(len(syllogisms))))
    # Generate a rank column that will be used to sort
    # the dataframe numerically
    df['Tm_Rank'] = df['syllog'].map(sorterIndex)
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

    return agg_answers


def generate_syllogisms():
    moods = ['AA', 'AI', 'IA', 'AE', 'EA', 'AO', 'OA', 'II',
             'IE', 'EI', 'IO', 'OI', 'EE', 'EO', 'OE', 'OO']
    syllogisms = []
    for mood in moods:
        for figure in ('1', '2', '3', '4'):
            syllogisms += [mood + figure]
    return(syllogisms)


def populate_queue(part_data):
    '''
    Insert data into queue for parallel execution.
    '''
    part_names = sorted(list(part_data.keys()))

    # Initialize a Queue object for queueing the data
    request_queue = mp.Queue()

    # For every participant, get her data
    for part in part_names:
        out_name = './traces/' + part + '.pickle'

        # Check if trace already exists
        if os.path.exists(out_name):
            continue

        # Put data in queue
        data = part_data[part]
        request_queue.put((out_name, data))

    return request_queue


def manage_processes(processes, request_queue, max_processes=2):
    '''
    Delete dead processes and start new ones.
    '''

    for i in range(0, len(processes)):
        if processes[i].is_alive() is not True:
            print(processes[i])
            processes[i].terminate()
            del(processes[i])

        if len(processes) < max_processes:
            w = Worker(request_queue, n_init=N_INIT, draws=DRAWS)
            w.start()
            processes.append(w)

    return processes


class Worker(mp.Process):
    # Convergence Parameters

    def __init__(self, queue, draws=20000, n_init=200000):
        super(Worker, self).__init__()
        self.queue = queue
        self.draws = draws
        self.n_init = n_init

    def run(self):
        print('Worker started')
        # do some initialization here

        for out_name, data in iter(self.queue.get, None):
            print('Computing: {}'.format(out_name))
            # get participant number
            n = data.iloc[0].sum()
            model, trace = phm.get_posterior(data=data, n=n, draws=self.draws,
                                             n_init=self.n_init,
                                             progressbar=False)
            with open(out_name, 'wb') as pickle_file:
                pickle.dump({'model': model, 'trace': trace}, pickle_file,
                            pickle.HIGHEST_PROTOCOL)

            # Free memory
            model, trace, pickle_file = None, None, None
            print('Finished: {}'.format(out_name))
            return


if __name__ == '__main__':
    syllogisms = generate_syllogisms()

    # Parse and preprocess data
    file_name = '../data/rg16/rg16.csv'
    rg_data = get_rg_data(file_name=file_name)
    print('Data imported')

    # Aggregate data
    rg_df = pd.DataFrame(rg_data)
    rg_df.to_csv('rg_all.csv', sep=',', header=False, index=False, decimal='.')

    ex_data = get_hattori_data()

    # Add aggregated 139 data to data sets
    ex_data['rg16'] = rg_df

    # Add data to the queue
    request_queue = populate_queue(ex_data)

    N_INIT = 200 * 1000
    DRAWS = 500 * 1000
    processes = []
    for i in range(3):
        w = Worker(request_queue, n_init=N_INIT, draws=DRAWS)
        w.start()
        processes.append(w)

    while not request_queue.empty():
            manage_processes(processes, request_queue, max_processes=3)
            time.sleep(5)

    # Sentinel objects to allow clean shutdown: 1 per worker.
    for i in range(4):
        request_queue.put(None)
