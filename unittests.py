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

unittest.main()
