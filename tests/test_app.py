"""Drives the Streamlit app with AppTest; generate and unload_pipeline are fakes, so no GPU is needed.

The inpainting tab's drawable canvas can't be driven by AppTest, so only its empty state is checked.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from sd2 import generate as gen

MAIN = Path(__file__).resolve().parent.parent / "main.py"


@pytest.fixture
def fakes(monkeypatch):
    calls = SimpleNamespace(generate=[], unload=0)

    def fake_generate(prompt, pipeline_name, on_progress=None, seed=None, **kwargs):
        calls.generate.append(dict(prompt=prompt, pipeline_name=pipeline_name, seed=seed, **kwargs))
        if on_progress:
            on_progress(0, 2)
        return SimpleNamespace(image=Image.new("RGB", (64, 64), "blue"), seed=123 if seed is None else seed)

    def fake_unload():
        calls.unload += 1

    # main.py imports these by name each time the script runs, so patching the module works.
    monkeypatch.setattr(gen, "generate", fake_generate)
    monkeypatch.setattr(gen, "unload_pipeline", fake_unload)
    return calls


def run_app():
    at = AppTest.from_file(str(MAIN), default_timeout=30).run()
    assert not at.exception
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_three_tabs_render(fakes):
    at = run_app()
    assert [t.label for t in at.tabs] == ["Text to Image (txt2img)", "Inpainting", "Image to image (img2img)"]
    assert button(at, "Generate image")  # txt2img; the other tabs need an uploaded image


def test_txt2img_generate_shows_image_and_seed(fakes):
    at = run_app()
    button(at, "Generate image").click()
    at.run()
    assert not at.exception
    assert len(fakes.generate) == 1
    call = fakes.generate[0]
    assert call["pipeline_name"] == "txt2img" and call["seed"] is None
    assert [c.value for c in at.caption if c.value.startswith("Seed:")] == ["Seed: 123"]
    # AppTest has no image element; the sidebar's "Latest Output Image" is set from the result.
    assert not any("No output generated yet" in m.value for m in at.sidebar.markdown)


def test_given_seed_is_passed_through(fakes):
    at = run_app()
    at.text_input(key="txt2img-seed").set_value("42").run()
    button(at, "Generate image").click()
    at.run()
    assert fakes.generate[0]["seed"] == 42
    assert any(c.value == "Seed: 42" for c in at.caption)


@pytest.mark.parametrize("seed", ["abc", "-1", "1.5", str(2**64)])
def test_invalid_seed_shows_error_and_does_not_generate(fakes, seed):
    at = run_app()
    at.text_input(key="txt2img-seed").set_value(seed).run()
    assert [e.value for e in at.error] == ["The seed must be a whole number from 0 to 2**64-1"]
    assert not any(b.label == "Generate image" for b in at.button)
    assert fakes.generate == []


def test_unload_button_calls_unload_pipeline(fakes):
    at = run_app()
    button(at, "Unload model (frees GPU and system memory)").click()
    at.run()
    assert not at.exception
    assert fakes.unload == 1
