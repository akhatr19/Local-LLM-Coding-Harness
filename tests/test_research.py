import httpx
import pytest

from local_llm_harness.config import SearxNGSettings
from local_llm_harness.research import ResearchError, SearxNGClient


def settings(**overrides) -> SearxNGSettings:
    return SearxNGSettings(
        base_url="http://searxng.test",
        timeout_seconds=1,
        result_limit=2,
        fetch_result_limit=1,
        max_fetch_bytes=1024,
        **overrides,
    )


@pytest.mark.asyncio
async def test_searxng_json_search_preserves_urls_and_bounds_fetches() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "searxng.test":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Parser docs",
                            "url": "https://docs.example.test/parser?version=1",
                            "content": "Official parser behavior.",
                        },
                        {
                            "title": "Second result",
                            "url": "https://other.example.test/guide",
                            "content": "Additional guidance.",
                        },
                    ]
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=utf-8"},
            content=b"x" * 2048,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SearxNGClient(settings(), client=http_client)
        sources = await client.search_and_fetch(["parser api"])

    assert [source.url for source in sources] == [
        "https://docs.example.test/parser?version=1",
        "https://other.example.test/guide",
    ]
    assert len(sources[0].content.encode()) == 1024
    assert sources[1].content == ""
    assert len(requests) == 2
    assert requests[0].url.params["format"] == "json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, json={"unexpected": []}),
    ],
)
async def test_search_outage_and_malformed_response_are_errors(response: httpx.Response) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response)
    ) as http_client:
        client = SearxNGClient(settings(), client=http_client)
        with pytest.raises(ResearchError, match="SearXNG"):
            await client.search_and_fetch(["parser api"])


@pytest.mark.asyncio
async def test_private_result_urls_are_not_fetched() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Metadata", "url": "http://127.0.0.1/secrets", "content": "x"}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SearxNGClient(settings(), client=http_client)
        with pytest.raises(ResearchError, match="no safe research results"):
            await client.search_and_fetch(["unsafe"])


@pytest.mark.asyncio
async def test_result_redirects_are_not_followed() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host or "")
        if request.url.host == "searxng.test":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Redirecting result",
                            "url": "https://docs.example.test/redirect",
                            "content": "Safe search snippet.",
                        }
                    ]
                },
            )
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secrets"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as http_client:
        client = SearxNGClient(settings(), client=http_client)
        sources = await client.search_and_fetch(["redirect"])

    assert requested_hosts == ["searxng.test", "docs.example.test"]
    assert sources[0].content == ""
    assert sources[0].fetch_error == "research result redirects are not followed"
