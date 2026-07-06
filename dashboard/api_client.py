"""
Thin wrapper around the CartIQ REST API. The dashboard only ever talks to
the API -- never touches the database directly -- the same boundary a real
brand's dashboard would respect (TRD S2.1: "JWT token contains brand_id
claim; all queries filtered by JWT claim at gateway"; API-key auth plays
that role here).
"""
import requests


class CartIQClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-CartIQ-Key": api_key}

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(f"{self.base_url}{path}", headers=self.headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def whoami(self) -> dict:
        return self._get("/v1/me")

    def overview(self, brand_id: str) -> dict:
        return self._get(f"/v1/brands/{brand_id}/analytics/overview")

    def funnel(self, brand_id: str) -> list:
        return self._get(f"/v1/brands/{brand_id}/analytics/funnel")

    def channels(self, brand_id: str) -> list:
        return self._get(f"/v1/brands/{brand_id}/analytics/channels")

    def revenue(self, brand_id: str) -> dict:
        return self._get(f"/v1/brands/{brand_id}/analytics/revenue")

    def discounts_summary(self, brand_id: str) -> dict:
        return self._get(f"/v1/brands/{brand_id}/analytics/discounts")

    def segments(self, brand_id: str) -> dict:
        return self._get(f"/v1/brands/{brand_id}/segments")

    def sessions(self, brand_id: str, page: int = 1, page_size: int = 50) -> dict:
        return self._get(f"/v1/brands/{brand_id}/sessions", params={"page": page, "page_size": page_size})
