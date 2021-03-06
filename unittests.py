import unittest
import download

class TestImageDownloader(unittest.TestCase):
    def test_pathify(self):
        res = download._pathify("Squirrel & Moose\n")
        self.assertEqual(res, "Squirrel___Moose")


unittest.main()
