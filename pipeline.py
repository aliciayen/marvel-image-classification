import download
import imgfilter
import classifier

def run_group(cfgspec, log_fn):
    ''' pipeline.run_group(cfgspec, log_fn)

    Generates a list containing valid permutations of input parameters,
    according to the rules in 'cfgspec', and executes a run through the
    pipeline for each. The metrics of each run are stored by the
    callback function 'log_fn'.

    The configuration specification 'cfgspec' is a dictionary with the
    following format:

        parameters:
          dataset_filename: [100marvelcharacters.csv, kaggle.csv]
          base_search_term: "Marvel Comic Character"
          search_options:
              style: [None, lineart]
              domain: [None]
          download_options:
              count: 50
          optimizers:
            Adam:
              lr: [0.0001, 0.001, 0.01, 0.1]
            SGD:
              lr: [0.0001, 0.001, 0.01, 0.1]
              momentum: [0.9, 0.8, 0.7]
          test_size: 0.3

        suppress_permutation:
          - [optimizers.SGD.lr, optimizers.SGD.momentum]

    Each key under 'parameters' corresponds to a parameter passed to
    pipeline.run_pass(), but the values are instead a list of values to
    be tested. All possible permutations of the values given for the
    parameters will be added to the test list, unless explicitly
    suppressed by an entry in 'suppress_permutation', which is a list of
    pairs of dot-delimited keys that are independent, and need not be
    varied against each other.
    '''

    raise NotImplementedError()

def run_pass(cfg, imagecache='images/'):
    ''' pipeline.run_pass(cfg) -> metrics

    Given a set of parameters, 'cfg', of the format defined below, run
    a single pass through the pipeline. This includes loading a dataset,
    downloading the images, filtering undesirable images, training the
    CNN classifier, and producing the resulting scoring metrics. Returns
    a dictionary containing the scoring metrics.

    The format of the input parameters is a dictionary. An example of
    the structure is given below:

        'dataset_filename': '100marvelcharacters.csv',
        'base_search_term': 'Marvel Comic Character',
        'search_options': {'style': 'lineart'},
        'optimizer': ('SGD', {'lr': 0.0001, 'momentum': 0.9}),
        'test_size': 0.3,

    'dataset_filename' is the path to the dataset to load.

    'base_search_term' is a string appended to the character name when
    executing the Google image search.

    'search_options' is the kwargs dictionary passed to
    download.generate_search_url().

    'optimizer' is a tuple containing the name of the optimizer in
    torch.optim, and the kwargs dictionary passed to its constructor.

    'test_size' is the fraction of the data to hold out from the
    training set for validation.
    '''

    raise NotImplementedError()
