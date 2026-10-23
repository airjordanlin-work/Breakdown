import logging

from api.server import _QuietPollFilter


def _record(path, status):
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0,
        '%s - "%s %s HTTP/%s" %d', ("1.2.3.4:5", "GET", path, "1.1", status), None,
    )


def test_successful_polls_are_dropped():
    f = _QuietPollFilter()
    assert not f.filter(_record("/imu/status", 200))
    assert not f.filter(_record("/sessions/latest?x=1", 200))


def test_other_requests_and_errors_are_kept():
    f = _QuietPollFilter()
    assert f.filter(_record("/moves", 200))
    assert f.filter(_record("/imu/status", 401))
