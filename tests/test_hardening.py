import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hardening.alerts import AlertNotifier, get_notifier
from hardening.secrets import SecretsVault


class TestSecretsVault(unittest.TestCase):
    def test_precedence_env_over_file_over_docker_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vault_file = tmp_path / "vault.json"
            docker_dir = tmp_path / "docker-secrets"
            docker_dir.mkdir(parents=True, exist_ok=True)

            vault_file.write_text(
                json.dumps(
                    {
                        "ALPACA_API_KEY": "from-file",
                        "NEWS_API_KEY": "from-file-news",
                    }
                ),
                encoding="utf-8",
            )
            (docker_dir / "ALPACA_API_KEY").write_text("from-docker", encoding="utf-8")
            (docker_dir / "ALPACA_SECRET").write_text("from-docker-secret", encoding="utf-8")

            vault = SecretsVault(
                file_path=str(vault_file),
                secrets_dir=str(docker_dir),
                environ={"ALPACA_API_KEY": "from-env"},
            )

            self.assertEqual(vault.get("ALPACA_API_KEY"), "from-env")
            self.assertEqual(vault.get("NEWS_API_KEY"), "from-file-news")
            self.assertEqual(vault.get("ALPACA_SECRET"), "from-docker-secret")

    def test_required_secret_raises(self) -> None:
        vault = SecretsVault(environ={})
        with self.assertRaises(EnvironmentError):
            vault.get("MISSING", required=True)


class TestAlertNotifier(unittest.TestCase):
    def test_disabled_notifier_is_noop(self) -> None:
        notifier = AlertNotifier(webhook_url="https://example.com", enabled=False)
        self.assertFalse(notifier.send("info", "Title", "Message"))

    @patch("hardening.alerts.requests.post")
    def test_webhook_payload_is_sent(self, post_mock: Mock) -> None:
        resp = Mock()
        resp.raise_for_status.return_value = None
        post_mock.return_value = resp

        notifier = AlertNotifier(webhook_url="https://example.com/hook", enabled=True, timeout_seconds=2)
        sent = notifier.send("warning", "Risk Gate", "Blocked", {"reason": "drawdown"})

        self.assertTrue(sent)
        post_mock.assert_called_once()
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["timeout"], 2)
        self.assertEqual(kwargs["json"]["title"], "Risk Gate")
        self.assertEqual(kwargs["json"]["details"]["reason"], "drawdown")

    def test_get_notifier_uses_vault_settings(self) -> None:
        vault = SecretsVault(
            environ={
                "ALERTS_ENABLED": "true",
                "ALERT_WEBHOOK_URL": "https://example.com/hook",
                "ALERT_TIMEOUT_SECONDS": "9",
            }
        )
        notifier = get_notifier(vault)
        self.assertTrue(notifier.enabled)
        self.assertEqual(notifier.webhook_url, "https://example.com/hook")
        self.assertEqual(notifier.timeout_seconds, 9.0)


if __name__ == "__main__":
    unittest.main()
