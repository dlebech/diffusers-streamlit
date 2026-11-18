import asyncio
import base64
import gc
import io
import json
import os
import sys
import weakref

import pytest
import torch
from mcp import Client
from PIL import Image

import mcp_server
from conftest import FakePipeline
from sd2 import generate as gen


class InMemoryClient:
    """Calls the server through the SDK's in-memory transport, one connection per call
    (an async fixture holding a connection open trips over pytest-asyncio's task handling)."""

    async def call_tool(self, name, arguments, **kwargs):
        async with Client(mcp_server.mcp) as c:
            return await c.call_tool(name, arguments, **kwargs)


@pytest.fixture
def client(fake_pipe):
    return InMemoryClient()


def parse(result):
    """Splits a tool result into (info dict, thumbnail PIL image)."""
    assert not result.is_error, result.content
    text = next(c for c in result.content if c.type == "text")
    image = next(c for c in result.content if c.type == "image")
    assert image.mime_type == "image/png"
    return json.loads(text.text), Image.open(io.BytesIO(base64.b64decode(image.data)))


def error_text(result):
    assert result.is_error
    return result.content[0].text


def make_image(path, size=(128, 96), color="red"):
    Image.new("RGB", size, color).save(path)
    return str(path)


async def test_list_models(client):
    result = await client.call_tool("list_models", {})
    models = {}
    for content in result.content:
        info = json.loads(content.text)
        models[info["slug"]] = info
    assert list(models) == ["z-image-turbo", "flux2-klein-4b", "sdxl", "sdxl-turbo", "sdxl-refiner"]
    z = models["z-image-turbo"]
    assert z["name"] == "Z-Image Turbo" and z["repo"] == "Tongyi-MAI/Z-Image-Turbo"
    assert z["tools"] == ["generate_image", "edit_image", "inpaint"]
    assert (z["default_steps"], z["default_guidance_scale"]) == (9, 0.0)
    assert models["sdxl-turbo"]["tools"] == ["generate_image"]
    assert (models["sdxl-turbo"]["default_width"], models["sdxl-turbo"]["default_height"]) == (512, 512)
    assert models["sdxl-refiner"]["tools"] == ["edit_image"]
    assert models["sdxl-refiner"]["modes"] == ["img2img"]
    assert models["sdxl"]["tools"] == ["generate_image", "inpaint"]
    assert models["sdxl"]["default_steps"] == 20 and models["sdxl"]["default_guidance_scale"] == 7.5
    assert "Turbo" in models["sdxl-turbo"]["hint"]


async def test_generate_image_uses_model_defaults(client, fake_pipe):
    info, thumb = parse(await client.call_tool("generate_image", {"prompt": "a dog", "model": "sdxl-turbo", "seed": 5}))
    assert os.path.isabs(info["path"]) and os.path.exists(info["path"])
    assert os.path.exists(info["settings_path"])
    assert info["path"].startswith(mcp_server.OUTPUT_DIR)
    assert (info["width"], info["height"], info["steps"], info["guidance_scale"]) == (512, 512, 2, 0.0)
    assert info["seed"] == 5 and info["model"] == "sdxl-turbo" and info["mode"] == "txt2img"
    assert info["ignored"] == []
    assert max(thumb.size) <= 256 and thumb.size == (256, 256)
    assert fake_pipe.calls[0]["steps"] == 2
    assert "Seed: 5" in open(info["settings_path"]).read()


async def test_generate_image_default_model_and_overrides(client, fake_pipe):
    info, thumb = parse(
        await client.call_tool(
            "generate_image",
            {"prompt": "x", "width": 640, "height": 320, "steps": 3, "guidance_scale": 1.5, "negative_prompt": "ugly"},
        )
    )
    assert info["model"] == "z-image-turbo"
    assert (info["width"], info["height"], info["steps"], info["guidance_scale"]) == (640, 320, 3, 1.5)
    assert thumb.size == (256, 128)
    assert fake_pipe.loads == [("txt2img", gen.Z_IMAGE_TURBO, True)]  # offload auto-enabled on 16GB
    assert fake_pipe.calls[0]["negative_prompt"] == "ugly"
    assert 0 <= info["seed"] < 2**32  # random


async def test_seed_reproducible(client, fake_pipe):
    args = {"prompt": "x", "model": "sdxl-turbo", "seed": 99}
    (a, _), (b, _) = parse(await client.call_tool("generate_image", args)), parse(await client.call_tool("generate_image", args))
    assert [c["seed"] for c in fake_pipe.calls] == [99, 99]
    assert Image.open(a["path"]).tobytes() == Image.open(b["path"]).tobytes()


async def test_edit_image(client, fake_pipe, tmp_path):
    src = make_image(tmp_path / "in.png")
    info, thumb = parse(await client.call_tool("edit_image", {"image_path": src, "prompt": "make it blue", "model": "sdxl-refiner", "seed": 3}))
    assert info["mode"] == "img2img" and info["strength"] == 0.3
    assert (info["width"], info["height"]) == (128, 96)
    assert fake_pipe.calls[0]["strength"] == 0.3
    assert fake_pipe.calls[0]["image"].size == (128, 96)
    assert (info["steps"], info["guidance_scale"]) == (20, 7.5)
    assert os.path.exists(info["path"])


async def test_inpaint_mask_white_is_repaint_and_resized(client, fake_pipe, tmp_path):
    src = make_image(tmp_path / "in.png", (128, 96))
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, (0, 0, 32, 48))  # left half white
    mask.save(tmp_path / "mask.png")
    info, _ = parse(
        await client.call_tool("inpaint", {"image_path": src, "mask_path": str(tmp_path / "mask.png"), "prompt": "a hat", "model": "sdxl", "strength": 0.9})
    )
    sent = fake_pipe.calls[0]["mask_image"]
    assert sent.size == (128, 96) and sent.mode == "L"
    assert sent.getpixel((10, 10)) == 255 and sent.getpixel((120, 10)) == 0
    assert fake_pipe.calls[0]["strength"] == 0.9 and info["strength"] == 0.9


async def test_inpaint_default_strength_is_one(client, fake_pipe, tmp_path):
    src = make_image(tmp_path / "in.png")
    mask = make_image(tmp_path / "mask.png", color="white")
    parse(await client.call_tool("inpaint", {"image_path": src, "mask_path": mask, "prompt": "x", "model": "sdxl"}))
    assert fake_pipe.calls[0]["strength"] == 1.0


async def test_unknown_model(client):
    text = error_text(await client.call_tool("generate_image", {"prompt": "x", "model": "nope"}))
    assert "Unknown model 'nope'" in text and "z-image-turbo" in text and "sdxl-turbo" in text


@pytest.mark.parametrize(
    "tool, model, listed, not_listed",
    [
        ("generate_image", "sdxl-refiner", "z-image-turbo, flux2-klein-4b, sdxl, sdxl-turbo", "sdxl-refiner,"),
        ("edit_image", "sdxl-turbo", "z-image-turbo, flux2-klein-4b, sdxl-refiner", "sdxl-turbo,"),
        ("inpaint", "sdxl-refiner", "z-image-turbo, flux2-klein-4b, sdxl", "sdxl-refiner,"),
    ],
)
async def test_model_does_not_support_tool(client, tmp_path, tool, model, listed, not_listed):
    src = make_image(tmp_path / "in.png")
    args = {"prompt": "x", "model": model}
    if tool != "generate_image":
        args["image_path"] = src
    if tool == "inpaint":
        args["mask_path"] = src
    text = error_text(await client.call_tool(tool, args))
    assert f"'{model}' does not support {tool}" in text
    assert listed in text


async def test_missing_and_unreadable_files(client, tmp_path):
    good = make_image(tmp_path / "in.png")
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    base = {"prompt": "x"}
    assert "does not exist" in error_text(await client.call_tool("edit_image", {**base, "image_path": str(tmp_path / "nope.png")}))
    assert "Could not read" in error_text(await client.call_tool("edit_image", {**base, "image_path": str(bad)}))
    text = error_text(await client.call_tool("inpaint", {**base, "image_path": good, "mask_path": str(tmp_path / "nomask.png")}))
    assert "mask file does not exist" in text


@pytest.mark.parametrize("width, height", [(1000, 512), (512, 10), (4096, 512), (512, 520 + 1)])
async def test_invalid_size(client, width, height):
    text = error_text(await client.call_tool("generate_image", {"prompt": "x", "width": width, "height": height}))
    assert "Invalid" in text and "multiple of 16" in text


@pytest.mark.parametrize("args", [{"steps": 0}, {"guidance_scale": -1}, {"seed": -1}])
async def test_invalid_numbers(client, args):
    assert "Invalid" in error_text(await client.call_tool("generate_image", {"prompt": "x", **args}))


async def test_invalid_strength(client, tmp_path):
    src = make_image(tmp_path / "in.png")
    text = error_text(await client.call_tool("edit_image", {"prompt": "x", "image_path": src, "strength": 1.5}))
    assert "Invalid strength" in text


async def test_out_of_memory_unloads_and_reports(client, monkeypatch):
    # Only a weakref to the pipeline survives outside generate()'s frame.
    refs = []
    alive_at_unload = []

    def fake_get_pipeline(name, version="Z-Image Turbo", enable_cpu_offload=False):
        pipe = FakePipeline()
        pipe.fail = torch.cuda.OutOfMemoryError("CUDA out of memory")
        refs.append(weakref.ref(pipe))
        return pipe

    def fake_unload():
        gc.collect()
        alive_at_unload.append(refs[-1]() is not None)

    monkeypatch.setattr(gen, "get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(gen, "unload_pipeline", fake_unload)
    text = error_text(await client.call_tool("generate_image", {"prompt": "x", "model": "sdxl-turbo"}))
    assert "out of memory" in text and "smaller" in text
    assert alive_at_unload == [False]  # the pipeline is really freed when unloaded


async def test_progress_notifications(client):
    progress = []

    async def on_progress(value, total, message):
        progress.append((value, total))

    result = await client.call_tool(
        "generate_image", {"prompt": "x", "model": "sdxl-turbo", "steps": 4}, progress_callback=on_progress
    )
    parse(result)
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


async def test_concurrent_calls_are_serialized(client, fake_pipe):
    fake_pipe.delay = 0.05
    args = {"prompt": "x", "model": "sdxl-turbo", "steps": 3}
    results = await asyncio.gather(*(client.call_tool("generate_image", {**args, "seed": i}) for i in range(3)))
    infos = [parse(r)[0] for r in results]
    assert sorted(i["seed"] for i in infos) == [0, 1, 2]
    assert fake_pipe.max_active == 1
    assert len(fake_pipe.calls) == 3


async def test_progress_total_is_the_real_number_of_steps(client):
    progress = []

    async def on_progress(value, total, message):
        progress.append((value, total))

    path = os.path.join(mcp_server.OUTPUT_DIR, "in.png")
    os.makedirs(mcp_server.OUTPUT_DIR, exist_ok=True)
    make_image(path)
    # 10 steps at strength 0.5 run 5.
    parse(await client.call_tool(
        "edit_image", {"image_path": path, "prompt": "x", "steps": 10, "strength": 0.5}, progress_callback=on_progress
    ))
    assert progress == [(i, 5) for i in range(1, 6)]


@pytest.mark.parametrize("tool", ["edit_image", "inpaint"])
@pytest.mark.parametrize("size", [(10, 10), (30, 200), (4000, 128)])
async def test_edit_rejects_bad_input_size(client, fake_pipe, tmp_path, tool, size):
    path = str(tmp_path / "in.png")
    make_image(path, size=size)
    args = {"image_path": path, "prompt": "x"}
    if tool == "inpaint":
        make_image(str(tmp_path / "m.png"), size=size)
        args["mask_path"] = str(tmp_path / "m.png")
    assert "between 64 and 2048" in error_text(await client.call_tool(tool, args))
    assert fake_pipe.calls == []


async def test_sdxl_inpaint_only_checks_the_minimum_input_size(client, fake_pipe, tmp_path):
    big = str(tmp_path / "big.png")
    mask = str(tmp_path / "m.png")
    make_image(big, size=(3000, 2000))
    make_image(mask, size=(3000, 2000))
    args = {"image_path": big, "mask_path": mask, "prompt": "x", "model": "sdxl"}
    parse(await client.call_tool("inpaint", args))
    small = str(tmp_path / "small.png")
    make_image(small, size=(10, 10))
    text = error_text(await client.call_tool("inpaint", {**args, "image_path": small}))
    assert "between 64 and 2048" in text
