import os
import shutil
import unittest
import download
import pipeline
import pandas
import yaml

class TestImageDownloader(unittest.TestCase):
    def test_pathify(self):
        res = download.pathify("Squirrel & Moose\n")
        self.assertEqual(res, "Squirrel_Moose")

    def test_google_url_gen(self):
        url = download.generate_search_url('wibble')
        baseurl, params = url.split('?')

        self.assertEqual(baseurl, "https://www.google.com/search")

        params_list = params.split('&')
        if 'tbm=isch' not in params_list:
            self.fail("Image search type not set appropriately")
        if 'q=wibble' not in params_list:
            self.fail("Search keyword not set appropriately")

    def test_image_style(self):
        url = download.generate_search_url("wibble", style='clipart')
        baseurl, params = url.split('?')

        self.assertEqual(baseurl, "https://www.google.com/search")

        params_list = params.split('&')
        if 'tbm=isch' not in params_list:
            self.fail("Image search type not set appropriately")
        if 'q=wibble' not in params_list:
            self.fail("Search keyword not set appropriately")
        if 'tbs=itp%3Aclipart' not in params_list:
            self.fail("Image style not set appropriately")

        self.assertRaises(ValueError, download.generate_search_url,
                           "wibble", style='broken')

    def test_domain_filter(self):
        url = download.generate_search_url("wibble", domain='example.com')
        baseurl, params = url.split('?')

        self.assertEqual(baseurl, "https://www.google.com/search")

        params_list = params.split('&')
        if 'q=wibble+site%3Aexample.com' not in params_list:
            self.fail("Domain filtering not set appropriately")

    def test_download(self):
        dl_dir = "./unittest-images/"

        if os.path.exists(dl_dir):
            shutil.rmtree('./unittest-images/')

        os.mkdir('./unittest-images/')
        try:
            url = download.generate_search_url("squirrel")
            download.download_images(url, './unittest-images/squirrel', 3)
            for i in range(0, 3):
                paths = (
                    './unittest-images/squirrel.%03i.jpeg' % i,
                    './unittest-images/squirrel.%03i.png' % i,
                )
                for path in paths:
                    if not os.path.exists(path):
                        break
                else:
                    self.fail("Image file '%s' does not exist" % path)
        finally:
            shutil.rmtree('./unittest-images/')

    def test_start_offset(self):
        url = 'https://www.google.com/search?q=test&tbm=isch&tbs='
        exp = 'https://www.google.com/search?q=test&tbm=isch&start=42'
        new = download._set_search_url_offset(url, 42)
        self.assertEqual(new, exp)

        url = 'https://www.google.com/search?q=test&tbm=isch&tbs=a'
        exp = 'https://www.google.com/search?q=test&tbm=isch&tbs=a&start=42'
        new = download._set_search_url_offset(url, 42)
        self.assertEqual(new, exp)


class TestPipeline(unittest.TestCase):
    def test_hash(self):
        dataset1 = pandas.DataFrame([
            {"Character": "Good Guy", "Label": "Hero"},
            {"Character": "Bad Guy", "Label": "Villain"},
        ])

        dataset2 = pandas.DataFrame([
            {"Character": "Gooder Guy", "Label": "Hero"},
            {"Character": "Bad Guy", "Label": "Villain"},
        ])

        h1 = pipeline.generate_hash(dataset1, "Marvel Character", {})
        h2 = pipeline.generate_hash(dataset2, "Marvel Character", {})
        self.assertNotEqual(h1, h2)

        h1 = pipeline.generate_hash(dataset1, "", {})
        h2 = pipeline.generate_hash(dataset1, "Marvel Comic Character", {})
        self.assertNotEqual(h1, h2)

        h1 = pipeline.generate_hash(dataset1, "", {})
        h2 = pipeline.generate_hash(dataset1, "", {'style': 'lineart'})
        self.assertNotEqual(h1, h2)

    def test_permutation(self):
        cfgspec = {
            'parameters': {
                'dataset_filename': ['100marvelcharacters.csv', 'kaggle.csv'],
                'base_search_term': [
                    'Marvel Comic Character',
                    'Character No Background'
                ],
                'search_style': ['None', 'lineart'],
                'search_domain': ['None'],
                'optimizer': {
                    'Adam': {'lr': [0.0001, 0.001, 0.01, 0.1]},
                    'SGD': {
                        'lr': [0.0001, 0.001, 0.01, 0.1],
                        'momentum': [0.9, 0.8, 0.7]
                    }
                },
                'test_size': [0.3],
                'val_size': [0.1]
            }
        }

        res = pipeline._get_permutations(cfgspec)

        # As written, the example spec should have:
        # 2 * 2 * 2 * (4 + 4*3) = 128 permutations
        self.assertEqual(len(res), 128)

    def test_batch(self):
        logdata = []
        def do_log(params, results):
            logdata.append((params, results))

        dl_dir = "./unittest-images/"
        cfgspec = {
            'parameters': {
                'dataset_filename': ['100marvelcharacters.csv'],
                'base_search_term': [''],
                'search_style': [None],
                'search_domain': [None],
                'optimizer': {
                    'SGD': {
                        'lr': [0.1],
                        'momentum': [0.9],
                     },
                     'Adam': {
                        'lr': [0.1],
                     }
                },
                'test_size': [0.3],
                'val_size': [0.1],
            }
        }

        pipeline.run_group(cfgspec, do_log, imagecache=dl_dir, n_images=2)

        optimizers = [p['optimizer'] for p, res in logdata]
        self.assertTrue(('SGD', {'lr': 0.1, 'momentum': 0.9}) in optimizers)
        self.assertTrue(('Adam', {'lr': 0.1}) in optimizers)

        for params, results in logdata:
            actual = sorted(results.keys())
            expected = sorted(('train', 'val', 'test'))
            self.assertEqual(actual, expected)

            for testtype in results.keys():
                actual = sorted(results[testtype].keys())
                expected = sorted(('loss', 'accuracy'))
                self.assertEqual(actual, expected)
                for key in expected:
                    self.assertTrue(isinstance(results[testtype][key], float))

    def test_single_pass(self):
        dl_dir = "./unittest-images/"
        config = {
            'dataset_filename': '100marvelcharacters.csv',
            'base_search_term': '',
            'search_options': {},
            'optimizer': ('Adam', {'lr': 0.0001}),
            'output_dir': 'images',
            'test_size': 0.3,
            'val_size': 0.1,
        }

        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)
        try:
            metrics = pipeline.run_pass(config, imagecache=dl_dir, n_images=2)
        finally:
            if os.path.exists(dl_dir):
                shutil.rmtree(dl_dir)

        self.assertEqual(set(['train', 'val', 'test']), set(metrics.keys()))

        for k in ('train', 'val', 'test'):
            loss = metrics[k]['loss']
            acc = metrics[k]['accuracy']

            self.assertTrue(isinstance(loss, float))
            self.assertTrue(0.0 < loss < 1.0)
            self.assertTrue(isinstance(acc, float))
            self.assertTrue(0.0 < acc < 1.0)



unittest.main()
