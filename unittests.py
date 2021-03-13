import os
import shutil
import unittest
import download

class TestImageDownloader(unittest.TestCase):
    def test_pathify(self):
        res = download._pathify("Squirrel & Moose\n")
        self.assertEqual(res, "Squirrel___Moose")

    def test_google_url_gen(self):
        url = download._generate_google_url('wibble')
        baseurl, params = url.split('?')

        self.assertEqual(baseurl, "https://www.google.com/search")

        params_list = params.split('&')
        if 'tbm=isch' not in params_list:
            self.fail("Image search type not set appropriately")
        if 'q=wibble' not in params_list:
            self.fail("Search keyword not set appropriately")

    def test_download(self):
        dl_dir = "./unittest-images/"

        if os.path.exists(dl_dir):
            shutil.rmtree('./unittest-images/')

        os.mkdir('./unittest-images/')
        try:
            download.download_images('google', 'squirrel',
                                     './unittest-images/squirrel', 3)
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


unittest.main()
