from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_dashboard_runs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].label == "Predicted profit"


def test_loss_and_extrapolation_flags():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.number_input[0].set_value(10_000).run()
    assert at.metric[0].value.startswith("-$")
    assert len(at.error) == 1
    at.number_input[0].set_value(900_000).run()
    assert len(at.warning) == 1
