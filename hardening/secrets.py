import json
import os
from pathlib import Path


class SecretsVault:
    """Resolve secrets from environment, optional JSON vault file, and Docker secrets."""

    def __init__(
        self,
        file_path: str | None = None,
        secrets_dir: str | None = None,
        environ: dict | None = None,
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._file_path = file_path or self._environ.get("SECRETS_VAULT_FILE", "")
        self._secrets_dir = Path(secrets_dir or self._environ.get("SECRETS_DIR", "/run/secrets"))
        self._file_cache: dict[str, str] | None = None

    def _load_file_cache(self) -> dict[str, str]:
        if self._file_cache is not None:
            return self._file_cache

        self._file_cache = {}
        if not self._file_path:
            return self._file_cache

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                self._file_cache = {str(k): str(v) for k, v in payload.items() if v is not None}
        except (OSError, json.JSONDecodeError):
            # Keep vault lookup non-fatal; runtime validation handles required keys.
            self._file_cache = {}

        return self._file_cache

    def _read_secret_file(self, name: str) -> str | None:
        candidates = [name, name.upper(), name.lower()]
        for candidate in candidates:
            secret_file = self._secrets_dir / candidate
            if not secret_file.exists() or not secret_file.is_file():
                continue
            try:
                return secret_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
        return None

    def get(self, name: str, default: str | None = None, required: bool = False) -> str | None:
        value = self._environ.get(name)
        if value is None:
            file_cache = self._load_file_cache()
            value = file_cache.get(name)
        if value is None:
            value = self._read_secret_file(name)

        if (value is None or value == "") and required:
            raise EnvironmentError(f"Missing required secret: {name}")
        if value in (None, ""):
            return default
        return value
