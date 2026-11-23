"""Keep benchmark credentials in memory; persist redacted audit evidence."""
import json
import re

SECRET_KEYS = {"password", "new_password", "old_password", "access_token",
               "refresh_token", "token", "authorization", "api_key"}


def redact(value):
    if isinstance(value, dict):
        return {key: "[redacted]" if key.lower() in SECRET_KEYS else redact(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            # Model prose may quote a JWT rather than return a structured field.
            return re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
                          "[redacted-jwt]", value)
        if isinstance(parsed, (dict, list)):
            return json.dumps(redact(parsed))
    return value
