import os
import re

import pytest

from sd2 import generate as gen
from tests.conftest import FakePipeline, NoNegativePromptPipeline


def run(tmp_path, **kwargs):
    args = dict(prompt="a cat", pipeline_name="txt2img", output_dir=str(tmp_path), version=gen.SD_XL_TURBO)
    args.update(kwargs)
    return gen.generate(args.pop("prompt"), args.pop("pipeline_name"), **args)


def test_does_not_import_streamlit():
    import subprocess, sys

    code = "import sys, sd2.generate; sys.exit('streamlit' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_slugs_map_to_pipelines():
    assert set(gen.MODEL_SLUGS) == {"z-image-turbo", "flux2-klein-4b", "sdxl", "sdxl-turbo", "sdxl-refiner"}
    versions = {version for (_, version) in gen.PIPELINES}
    assert set(gen.MODEL_SLUGS.values()) == versions


def test_seed_is_saved_and_returned(fake_pipe, tmp_path):
    result = run(tmp_path, seed=1234)
    assert result.seed == 1234
    assert fake_pipe.calls[0]["seed"] == 1234
    assert "Seed: 1234" in result.settings
    assert "Mode: txt2img" in result.settings
    assert "Seed: 1234" in open(result.path[:-4] + ".txt").read()


def test_random_seed_when_omitted(fake_pipe, tmp_path):
    a = run(tmp_path)
    b = run(tmp_path)
    assert a.seed != b.seed
    assert fake_pipe.calls[0]["seed"] == a.seed
    assert f"Seed: {a.seed}" in a.settings


def test_same_seed_same_image(fake_pipe, tmp_path):
    a = run(tmp_path, seed=7)
    b = run(tmp_path, seed=7)
    c = run(tmp_path, seed=8)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.image.tobytes() != c.image.tobytes()


def test_progress_callback(fake_pipe, tmp_path):
    seen = []
    run(tmp_path, steps=3, on_progress=lambda step, total: seen.append((step, total)))
    assert seen == [(0, 3), (1, 3), (2, 3)]


def test_unsupported_kwargs_are_ignored_and_reported(monkeypatch, tmp_path):
    pipe = NoNegativePromptPipeline(distilled=True)
    monkeypatch.setattr(gen, "get_pipeline", lambda *a, **k: pipe)
    monkeypatch.setattr(gen, "gpu_memory_bytes", lambda: 16 * 2**30)
    result = run(tmp_path, version=gen.FLUX_2_KLEIN_4B, negative_prompt="blurry", seed=1)
    assert "negative_prompt" in result.ignored
    assert "guidance_scale" in result.ignored  # distilled
    assert "Negative prompt: blurry (ignored by this model)" in result.settings
    assert "negative_prompt" not in pipe.calls[0] or pipe.calls[0]["negative_prompt"] is None


def test_no_ignored_when_nothing_unsupported(fake_pipe, tmp_path):
    assert run(tmp_path).ignored == []


def test_output_files_saved(fake_pipe, tmp_path):
    result = run(tmp_path, prompt="a  big\ncat", width=64, height=64)
    assert result.path.startswith(str(tmp_path))
    assert re.search(r"a_big_cat_[\d.]+\.png$", result.path)
    assert result.image.size == (64, 64)


def test_cpu_offload_depends_on_gpu_memory(monkeypatch):
    monkeypatch.setattr(gen, "gpu_memory_bytes", lambda: 16 * 2**30)
    assert gen.needs_cpu_offload(gen.Z_IMAGE_TURBO)
    assert not gen.needs_cpu_offload(gen.SD_XL_10)
    monkeypatch.setattr(gen, "gpu_memory_bytes", lambda: 48 * 2**30)
    assert not gen.needs_cpu_offload(gen.Z_IMAGE_TURBO)
    assert gen.needs_cpu_offload(gen.Z_IMAGE_TURBO, True)


def test_img2img_keeps_image_size(fake_pipe, tmp_path):
    from PIL import Image

    result = run(tmp_path, pipeline_name="img2img", version=gen.Z_IMAGE_TURBO, image_input=Image.new("RGB", (100, 70)), strength=0.5)
    # Sizes are rounded down to a multiple of 16 for the newer models.
    assert (fake_pipe.calls[0]["width"], fake_pipe.calls[0]["height"]) == (96, 64)
    assert "Strength: 0.5" in result.settings


def test_loader_keeps_one_pipeline(monkeypatch):
    loaded = []

    class Pipe:
        def __init__(self, repo):
            self.repo = repo

        def to(self, device):
            return self

    class Cls:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            loaded.append(repo)
            return Pipe(repo)

    monkeypatch.setitem(gen.PIPELINES, ("txt2img", gen.SD_XL_TURBO), (Cls, "turbo", {}))
    monkeypatch.setitem(gen.PIPELINES, ("txt2img", gen.SD_XL_10), (Cls, "base", {}))
    gen.unload_pipeline()
    a = gen.get_pipeline("txt2img", gen.SD_XL_TURBO)
    assert gen.get_pipeline("txt2img", gen.SD_XL_TURBO) is a
    assert loaded == ["turbo"]
    b = gen.get_pipeline("txt2img", gen.SD_XL_10)
    assert b is not a and gen._loaded[1] is b
    gen.unload_pipeline()
    assert gen._loaded is None


@pytest.mark.parametrize("prompt", ["AC/DC poster", "../escaped", "/etc/x", "...", "a\\b:c"])
def test_prompt_cannot_escape_output_dir(fake_pipe, tmp_path, prompt):
    out = tmp_path / "outputs"
    result = gen.generate(prompt, "txt2img", steps=1, width=64, height=64, output_dir=str(out))
    assert os.path.dirname(result.path) == str(out)
    assert os.path.exists(result.path) and os.path.exists(result.settings_path)
    assert os.listdir(tmp_path) == ["outputs"]
