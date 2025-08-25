import os
import random
from datetime import datetime, timedelta, timezone

from locust import HttpUser, task, between


class LinkUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = os.getenv("BASE_URL", "http://localhost:9999")

    @task(3)
    def create_links(self):
        alias = f"u{random.randint(100000, 999999)}"
        self.client.post(
            "/links/shorten",
            json={
                "original_url": f"https://example.com/{alias}",
                "custom_alias": alias,
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            },
            name="POST /links/shorten",
        )

    @task(2)
    def redirect_existing(self):
        # hit some static aliases that may or may not exist; errors still stress the system
        alias = random.choice(["demoalias", "a1", "a2", "newcode123", "missing"])
        self.client.get(f"/{alias}", name="GET /{short_code}")


