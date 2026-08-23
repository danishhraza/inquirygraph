from inquirygraph.tools.web_search import _is_private_host


def test_blocks_loopback_and_private_hosts():
    assert _is_private_host("127.0.0.1")
    assert _is_private_host("localhost")
    assert _is_private_host("10.0.0.5")
    assert _is_private_host("192.168.1.1")
    assert _is_private_host("169.254.169.254")  # cloud metadata endpoint


def test_allows_public_hosts():
    assert not _is_private_host("93.184.216.34")
    assert not _is_private_host("8.8.8.8")