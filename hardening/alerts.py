import logging
from datetime import datetime, timezone

import requests

from hardening.secrets import SecretsVault

logger = logging.getLogger(__name__)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class AlertNotifier:
    """Best-effort alert delivery via HTTP webhook."""

    def __init__(
        self,
        webhook_url: str | None,
        enabled: bool = False,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds

    def send(
        self,
        level: str,
        title: str,
        message: str,
        details: dict | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        payload = {
            "source": "trading-bot",
            "level": level.upper(),
            "title": title,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self.webhook_url:
            logger.warning("Alerts enabled but ALERT_WEBHOOK_URL is not configured.")
            return False

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.warning("Alert delivery failed: %s", exc)
            return False


def get_notifier(vault: SecretsVault | None = None) -> AlertNotifier:
    secret_vault = vault or SecretsVault()
    enabled = _as_bool(secret_vault.get("ALERTS_ENABLED", "false"), default=False)
    webhook = secret_vault.get("ALERT_WEBHOOK_URL")
    timeout_raw = secret_vault.get("ALERT_TIMEOUT_SECONDS", "5")
    try:
        timeout = float(timeout_raw) if timeout_raw is not None else 5.0
    except ValueError:
        timeout = 5.0
    return AlertNotifier(webhook_url=webhook, enabled=enabled, timeout_seconds=timeout)
