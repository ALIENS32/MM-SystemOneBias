"""Authentication and explicit HTTP error handling for API models."""

from __future__ import annotations


class ApiError(RuntimeError):
    """Raised when a remote API request fails or returns an invalid response."""


def api_client(config: dict, extra_headers: dict | None = None):
    import httpx

    key = config.get("api_key")
    if not key:
        raise RuntimeError(
            "Missing API credential: export the credential environment variable referenced by api_key in the model YAML"
        )
    base_url = config.get("base_url", "").strip()
    if not base_url:
        raise RuntimeError("Missing API base_url in the model YAML")
    headers = {
        "Authorization": f"Bearer {key}",
        **config.get("headers", {}),
        **(extra_headers or {}),
    }
    return httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        headers=headers,
        timeout=config.get("timeout_seconds", config.get("timeout", 180)),
    )


def post_json(client, endpoint: str, body: dict) -> dict:
    response = client.post(endpoint, json=body)
    if response.is_error:
        error = ApiError(f"API {endpoint} failed: HTTP {response.status_code}")
        error.retryable = response.status_code in {408, 425, 429} or response.status_code >= 500
        error.fatal = response.status_code in {400, 401, 403, 404, 422}
        raise error
    try:
        return response.json()
    except ValueError as error:
        content_type = response.headers.get("content-type", "unknown")
        preview = response.text[:500].replace("\n", "\\n")
        raise ApiError(
            f"API {endpoint} returned non-JSON content: HTTP {response.status_code}, "
            f"content-type={content_type}, body={preview}"
        ) from error
