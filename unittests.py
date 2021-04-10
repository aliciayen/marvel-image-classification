import os
import shutil
import unittest
import download
import pipeline
import pandas

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

    def test_single_pass(self):
        dl_dir = "./unittest-images/"
        config = {
            'dataset_filename': '100marvelcharacters.csv',
            'base_search_term': '',
            'search_options': {},
            'optimizer': ('Adam', {'lr': 0.0001}),
            'output_dir': 'images',
            'test_size': 0.3,
        }

        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)
        try:
            metrics = pipeline.run_pass(config, imagecache=dl_dir, n_images=2)
        finally:
            if os.path.exists(dl_dir):
                shutil.rmtree(dl_dir)

        self.assertEqual(set(['train', 'test']), set(metrics.keys()))

        for k in ('train', 'test'):
            loss = metrics[k]['loss']
            acc = metrics[k]['accuracy']

            self.assertTrue(isinstance(loss, float))
            self.assertTrue(0.0 < loss < 1.0)
            self.assertTrue(isinstance(acc, float))
            self.assertTrue(0.0 < acc < 1.0)


unittest.main()
