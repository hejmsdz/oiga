import unittest

from oiga.cli import CLI

class CLITest(unittest.TestCase):
    def test_run(self):
        cli = CLI()
        self.assertTrue(callable(cli.run))

if __name__ == '__main__':
    unittest.main()
