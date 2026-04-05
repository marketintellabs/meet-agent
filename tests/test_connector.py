"""Tests for meeting connector utilities."""

import pytest

from meet_agent.connector.base import MeetingConnector


def test_detect_google_meet():
    assert MeetingConnector.detect_platform("https://meet.google.com/abc-defg-hij") == "google_meet"


def test_detect_zoom():
    assert MeetingConnector.detect_platform("https://us05web.zoom.us/j/123456") == "zoom"
    assert MeetingConnector.detect_platform("https://zoom.us/j/123456") == "zoom"


def test_detect_teams():
    url = "https://teams.microsoft.com/l/meetup-join/abc"
    assert MeetingConnector.detect_platform(url) == "teams"


def test_detect_unknown():
    with pytest.raises(ValueError, match="Unsupported"):
        MeetingConnector.detect_platform("https://example.com/meeting")
