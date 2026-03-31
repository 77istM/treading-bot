import sqlite3
import unittest

from db.queries import read_bool_setting, read_setting, write_setting


class TestRuntimeSettings(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS settings "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )

    def tearDown(self) -> None:
        self.conn.close()

    def test_write_and_read_numeric_setting(self) -> None:
        write_setting(self.conn, "stop_loss_pct", 0.03)
        self.assertAlmostEqual(read_setting(self.conn, "stop_loss_pct", 0.01), 0.03)

    def test_write_and_read_bool_setting_true(self) -> None:
        write_setting(self.conn, "allow_crypto_shorts", True)
        self.assertTrue(read_bool_setting(self.conn, "allow_crypto_shorts", default=False))

    def test_write_and_read_bool_setting_false(self) -> None:
        write_setting(self.conn, "allow_stock_shorts", False)
        self.assertFalse(read_bool_setting(self.conn, "allow_stock_shorts", default=True))


if __name__ == "__main__":
    unittest.main()
