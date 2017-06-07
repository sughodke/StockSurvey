import unittest
from models.security import *


class SecurityTestHarness(Security):
    _today = datetime.date(2016, 12, 1)
    store_dir = '/tmp/'


class TestSyncMethods(unittest.TestCase):

    def setUp(self):
        self.security = SecurityTestHarness()

    def test_enddate(self):
        self.assertEqual(self.security.enddate,
                         SecurityTestHarness._today)
        self.assertEqual(self.security.daily.index[-1],
                         SecurityTestHarness._today)

    def test_nextday(self):
        self.security._today = datetime.date(2016, 12, 2)
        self.security.sync()
        self.assertEqual(self.security.daily.index[-1],
                         self.security._today)


class TestStoreMethods(unittest.TestCase):

    def setUp(self):
        self.security = SecurityTestHarness()

    def test_save(self):
        self.security.save()
        self.assertTrue(os.path.isfile(self.security._filename(self.security.ticker)),
                        True)

    def test_load(self):
        self.security.save()
        ticker = self.security.ticker
        loaded_security = SecurityTestHarness.load(ticker)
        self.assertEqual(self.security.daily.shape,
                         loaded_security.daily.shape)
        # loaded_security._today = datetime.date(2016, 12, 2)


if __name__ == '__main__':
    unittest.main()
