import gc
import os
import re
import weakref

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


def test_generate_trims_host_memory(fake_pipe, tmp_path, monkeypatch):
    trims = []
    monkeypatch.setattr(gen, "_trim_host_memory", lambda: trims.append(1))
    run(tmp_path)
    assert trims == [1]


def test_generate_trims_host_memory_when_it_fails(fake_pipe, tmp_path, monkeypatch):
    trims = []
    monkeypatch.setattr(gen, "_trim_host_memory", lambda: trims.append(1))
    fake_pipe.fail = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        run(tmp_path)
    assert trims == [1]


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


@pytest.fixture
def fake_loader(monkeypatch):
    """Fake pipeline classes for two modes and the same/another model; returns the event log."""
    events = []

    class Pipe:
        def __init__(self, source):
            self.source = source

        def to(self, device):
            events.append(("to", type(self).__name__, device))
            return self

        def remove_all_hooks(self):
            events.append(("remove_all_hooks", type(self).__name__))

        def enable_model_cpu_offload(self):
            events.append(("enable_model_cpu_offload", type(self).__name__))

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            events.append(("from_pretrained", cls.__name__, repo))
            return cls(repo)

        @classmethod
        def from_pipe(cls, pipeline, **kwargs):
            events.append(("from_pipe", cls.__name__, type(pipeline).__name__, kwargs))
            return cls(pipeline.source)

    class Txt2Img(Pipe):
        pass

    class Inpaint(Pipe):
        pass

    dtype = dict(dtype="bf16")
    monkeypatch.setitem(gen.PIPELINES, ("txt2img", gen.Z_IMAGE_TURBO), (Txt2Img, "z", dtype))
    monkeypatch.setitem(gen.PIPELINES, ("img2img", gen.Z_IMAGE_TURBO), (Txt2Img, "z", dtype))
    monkeypatch.setitem(gen.PIPELINES, ("inpaint", gen.Z_IMAGE_TURBO), (Inpaint, "z", dtype))
    monkeypatch.setitem(gen.PIPELINES, ("txt2img", gen.FLUX_2_KLEIN_4B), (Txt2Img, "flux", dtype))
    # Start and end with nothing loaded, even if a test fails.
    monkeypatch.setattr(gen, "_loaded", None)
    monkeypatch.setattr(gen, "_trim_host_memory", lambda: events.append(("trim",)))
    monkeypatch.setattr(gen, "unload_pipeline", lambda real=gen.unload_pipeline: (events.append(("unload",)), real()))
    return events


@pytest.mark.parametrize("offload", [True, False])
def test_mode_switch_reuses_components(fake_loader, offload):
    events = fake_loader
    old = weakref.ref(gen.get_pipeline("txt2img", gen.Z_IMAGE_TURBO, offload))
    events.clear()
    b = gen.get_pipeline("inpaint", gen.Z_IMAGE_TURBO, offload)
    assert type(b).__name__ == "Inpaint" and b.source == "z"
    assert gen._loaded == (("inpaint", gen.Z_IMAGE_TURBO, offload), b)
    assert ("remove_all_hooks", "Txt2Img") in events
    # The dtype is passed on, or from_pipe would cast the weights to float32.
    assert ("from_pipe", "Inpaint", "Txt2Img", {"dtype": "bf16"}) in events
    assert events.index(("remove_all_hooks", "Txt2Img")) < events.index(("from_pipe", "Inpaint", "Txt2Img", {"dtype": "bf16"}))
    assert not any(e[0] in ("from_pretrained", "unload") for e in events)
    last = ("enable_model_cpu_offload", "Inpaint") if offload else ("to", "Inpaint", "cuda")
    assert events[-1] == last
    # Only the new pipeline holds the components.
    gc.collect()
    assert old() is None


def test_mode_switch_same_class_returns_same_pipeline(fake_loader):
    events = fake_loader
    a = gen.get_pipeline("txt2img", gen.Z_IMAGE_TURBO, True)
    events.clear()
    assert gen.get_pipeline("img2img", gen.Z_IMAGE_TURBO, True) is a
    assert gen._loaded == (("img2img", gen.Z_IMAGE_TURBO, True), a)
    assert events == []


def test_model_or_offload_switch_reloads(fake_loader):
    events = fake_loader
    gen.get_pipeline("txt2img", gen.Z_IMAGE_TURBO, True)
    events.clear()
    b = gen.get_pipeline("txt2img", gen.FLUX_2_KLEIN_4B, True)
    assert events[:3] == [("unload",), ("trim",), ("from_pretrained", "Txt2Img", "flux")]
    events.clear()
    c = gen.get_pipeline("txt2img", gen.FLUX_2_KLEIN_4B, False)
    assert c is not b
    assert events[:3] == [("unload",), ("trim",), ("from_pretrained", "Txt2Img", "flux")]


def test_failed_mode_switch_leaves_nothing_loaded(fake_loader, monkeypatch):
    gen.get_pipeline("txt2img", gen.Z_IMAGE_TURBO, True)
    inpaint = gen.PIPELINES[("inpaint", gen.Z_IMAGE_TURBO)][0]

    def broken(pipeline, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(inpaint, "from_pipe", broken)
    with pytest.raises(RuntimeError, match="boom"):
        gen.get_pipeline("inpaint", gen.Z_IMAGE_TURBO, True)
    assert gen._loaded is None


@pytest.mark.parametrize("step", ["from_pipe", "enable_model_cpu_offload"])
def test_failed_mode_switch_frees_the_old_pipeline(fake_loader, monkeypatch, step):
    events = fake_loader
    old = weakref.ref(gen.get_pipeline("txt2img", gen.Z_IMAGE_TURBO, True))
    inpaint = gen.PIPELINES[("inpaint", gen.Z_IMAGE_TURBO)][0]
    real_from_pipe = inpaint.from_pipe

    def broken_from_pipe(pipeline, **kwargs):
        raise RuntimeError("boom")  # The traceback holds this frame, and so `pipeline`.

    def broken_offload(self):
        raise RuntimeError("boom")

    if step == "from_pipe":
        monkeypatch.setattr(inpaint, "from_pipe", broken_from_pipe)
    else:
        monkeypatch.setattr(inpaint, "from_pipe", real_from_pipe)
        monkeypatch.setattr(inpaint, "enable_model_cpu_offload", broken_offload)
    real_unload = gen.unload_pipeline

    def unload():
        # Before gc: only plain references would keep the old pipeline alive.
        events.append(("unload", old() is None))
        real_unload()

    monkeypatch.setattr(gen, "unload_pipeline", unload)
    events.clear()
    with pytest.raises(RuntimeError, match="boom") as info:
        gen.get_pipeline("inpaint", gen.Z_IMAGE_TURBO, True)
    assert gen._loaded is None
    # Unloaded (gc, CUDA cache, trim) after the old pipeline lost its references.
    assert ("unload", True) in events and events[-1] == ("trim",)
    assert ("unload", False) not in events
    # The traceback still says where it failed.
    assert info.traceback[-1].name in ("broken_from_pipe", "broken_offload")


@pytest.fixture
def fresh_malloc_trim():
    """Makes _trim_host_memory() look up malloc_trim again, before and after the test."""
    gen._malloc_trim.cache_clear()
    yield
    gen._malloc_trim.cache_clear()


def test_unload_trims_host_memory(monkeypatch, fresh_malloc_trim):
    trims = []

    class Libc:
        def malloc_trim(self, pad):
            trims.append(pad)

    monkeypatch.setattr(gen.ctypes, "CDLL", lambda name: Libc())
    gen.unload_pipeline()
    assert trims == [0]


def test_malloc_trim_is_looked_up_once(monkeypatch, fresh_malloc_trim):
    lookups, trims = [], []

    class Libc:
        def malloc_trim(self, pad):
            trims.append(pad)

    monkeypatch.setattr(gen.ctypes.util, "find_library", lambda name: lookups.append(name) or "libc.so.6")
    monkeypatch.setattr(gen.ctypes, "CDLL", lambda name: lookups.append(name) or Libc())
    for _ in range(3):
        gen._trim_host_memory()
    assert lookups == ["c", "libc.so.6"]
    assert trims == [0, 0, 0]


def test_missing_malloc_trim_is_harmless(monkeypatch, fresh_malloc_trim):
    names = []

    def libc_without_malloc_trim(name):
        names.append(name)
        return object()

    # libc without malloc_trim (e.g. macOS).
    monkeypatch.setattr(gen.ctypes, "CDLL", libc_without_malloc_trim)
    gen.unload_pipeline()
    assert len(names) == 1

    def no_libc(name):
        names.append(name)
        raise OSError("no libc")

    # No libc to be found (e.g. Windows): falls back to libc.so.6, which fails.
    gen._malloc_trim.cache_clear()
    monkeypatch.setattr(gen.ctypes, "CDLL", no_libc)
    monkeypatch.setattr(gen.ctypes.util, "find_library", lambda name: None)
    gen.unload_pipeline()
    assert names[-1] == "libc.so.6"
    assert gen._loaded is None


@pytest.mark.parametrize("prompt", ["AC/DC poster", "../escaped", "/etc/x", "...", "a\\b:c"])
def test_prompt_cannot_escape_output_dir(fake_pipe, tmp_path, prompt):
    out = tmp_path / "outputs"
    result = gen.generate(prompt, "txt2img", steps=1, width=64, height=64, output_dir=str(out))
    assert os.path.dirname(result.path) == str(out)
    assert os.path.exists(result.path) and os.path.exists(result.settings_path)
    assert os.listdir(tmp_path) == ["outputs"]
