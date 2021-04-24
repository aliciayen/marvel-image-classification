import os
import hashlib
import random
import pandas
import download
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

def run_pass(cfg, imagecache='images', n_images=100):
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

    dataset = pandas.read_csv(cfg['dataset_filename'])
    imgdir = prepare_imageset(dataset, cfg['base_search_term'],
                              cfg['search_options'], imagecache,
                              download_count=n_images)

    splitdir = train_test_split(imgdir, cfg['test_size'])

    opt_name, opt_kwargs = cfg['optimizer']
    stats_trn, stats_tst = classifier.evaluate(splitdir, opt_name, opt_kwargs)

    return {'train': stats_trn, 'test': stats_tst}

def prepare_imageset(dataset, base_search_term, search_opts, output_dir,
                     download_count=100):
    ''' pipeline.prepare_imageset(...) -> imagedir

    Downloads a base set of images corresponding to the entries in
    'dataset', if they're not already cached. Returns the directory
    containing the ImageFolder-structured tree of images.
    '''

    hashvalue = generate_hash(dataset, base_search_term, search_opts)
    imgdir = '%s/%s' % (output_dir, hashvalue)

    if not os.path.exists(imgdir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Create directory structure
        os.mkdir(imgdir)
        os.mkdir("%s/base/" % imgdir)
        for label in dataset["Label"].unique():
            os.mkdir("%s/base/%s" % (imgdir, label))

        # Download images
        for row in dataset.itertuples():
            name = row.Character
            search_term = "%s %s" % (base_search_term, name)
            pattern = "%s/base/%s/%s" % (imgdir, row.Label,
                                         download.pathify(name))
            url = download.generate_search_url(search_term, **search_opts)
            download.download_images(url, pattern, download_count)

        filter_imageset(imgdir)

    return imgdir

def filter_imageset(imagedir):
    ''' pipeline.filter_imageset(imagedir)

    Given a directory of images, delete images that are deemed
    "undesirable" using a CNN classifier.
    '''
    # FIXME: implement filtering
    return

def train_test_split(imagedir, test_fraction):
    ''' pipeline.train_test_split(imagedir, test_fraction) -> splitdir

    Splits an ImageFolder-structured directory into train and test
    datasets, creating a new directory at the same level as 'imagedir'.
    'imagedir' is the path to the directory containing the class
    directories, and 'test_fraction is a floating point value between
    0.0 and 1.0 that specifies the fraction of images to be placed in
    the test dataset (the remaining images go into the training
    dataset). Returns the path to the new directory containing the test/
    and train/ directories.
    '''
    # If we already have a directory with the split train/test data, don't
    # bother doing it again.
    splitdir = "%s/split-%0.2f" % (imagedir, test_fraction)
    if os.path.exists(splitdir):
        return splitdir

    # Logically split the entries into train/test data sets
    rng = random.Random()
    rng.seed(1337)
    classes = os.listdir("%s/base" % imagedir)
    test_list = set()
    train_list = set()
    for c in classes:
        class_imgs = os.listdir("%s/base/%s" % (imagedir, c))
        class_imgs = ["%s/%s" % (c, x) for x in class_imgs]
        n_test = int(len(class_imgs) * test_fraction)

        sampled = rng.sample(class_imgs, n_test)
        test_list = test_list.union(set(sampled))
        train_list = train_list.union(set(class_imgs) - test_list)

    # Create new directories for split data
    traindir = "%s/train" % splitdir
    testdir = "%s/test" % splitdir
    os.mkdir(splitdir)
    for d in [traindir, testdir]:
        os.mkdir(d)
        for c in classes:
            os.mkdir("%s/%s" % (d, c))

    # Populate the split directory. On POSIX systems create symbolic links
    # using os.symlink to save disk space. This won't be an option on Windows,
    # so in that case, fall back to using os.copy.
    if os.name == 'posix':
        do_copy = os.link
    else:
        do_copy = shutil.copy

    for setname, splitset in [('test', test_list), ('train', train_list)]:
        for fname in splitset:
            src = "%s/base/%s" % (imagedir, fname)
            dest = "%s/%s/%s" % (splitdir, setname, fname)
            do_copy(src, dest)

    return splitdir

def generate_hash(dataset, base_search_term, search_opts):
    ''' pipeline.generate_hash(data, base_search, search_opts) -> sha1

    Calculates a unique identifier (a SHA-1 cryptographic hash) from the
    input dataset, the base search term, and the search options. This
    is an easy and reproducible mechanism to detect when one of these
    values has changed, and consequently, when the images need to be
    downloaded again.
    '''

    hash_input = ""
    hash_input += str(base_search_term)

    for col in dataset:
        hash_input += str(col)
        for row in dataset[col]:
            hash_input += str(row)

    for k in sorted(search_opts.keys()):
        hash_input += "%s:%s" % (k, search_opts[k])

    hash_str = hashlib.sha1(bytes(hash_input, 'utf-8')).hexdigest()
    return hash_str
