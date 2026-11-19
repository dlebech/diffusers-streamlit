import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import mcp_server
from sd2 import generate as gen


class FakePipeline:
    """Stands in for a diffusers pipeline: explicit __call__ signature, returns .images."""

    def __init__(self, delay=0.0, distilled=False, fail=None):
        self.config = SimpleNamespace(is_distilled=distilled)
        self.calls = []
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def enable_attention_slicing(self):
        pass

    def disable_attention_slicing(self):
        pass

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        callback_on_step_end=None,
        guidance_scale=7.5,
        generator=None,
        width=64,
        height=64,
        image=None,
        mask_image=None,
        strength=1.0,
    ):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(
                dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=generator.initial_seed(),
                    width=width,
                    height=height,
                    image=image,
                    mask_image=mask_image,
                    strength=strength,
                )
            )
            if self.fail:
                raise self.fail
            # Like diffusers, img2img/inpaint only run steps * strength steps.
            n = max(1, int(num_inference_steps * strength)) if image is not None else num_inference_steps
            self._num_timesteps = n  # Like diffusers; scheduler.timesteps keeps its full length.
            for step in range(n):
                time.sleep(self.delay)
                if callback_on_step_end:
                    callback_on_step_end(self, step, None, {})
            size = image.size if image is not None else (width, height)
            # The pixels depend on the seed, like the real thing.
            value = torch.randint(0, 256, (3,), generator=generator).tolist()
            return SimpleNamespace(images=[Image.new("RGB", size, tuple(value))])
        finally:
            with self._lock:
                self.active -= 1


class NoNegativePromptPipeline(FakePipeline):
    """Like FLUX.2: no negative_prompt parameter."""

    def __call__(self, prompt, num_inference_steps=4, callback_on_step_end=None, guidance_scale=1.0, generator=None, width=64, height=64, image=None, strength=1.0):
        return super().__call__(
            prompt, None, num_inference_steps, callback_on_step_end, guidance_scale, generator, width, height, image, None, strength
        )


@pytest.fixture
def fake_pipe(monkeypatch, tmp_path):
    """Replaces the model loader with a FakePipeline; returns it. No GPU needed."""
    pipe = FakePipeline()
    loads = []

    def fake_get_pipeline(name, version="Z-Image Turbo", enable_cpu_offload=False):
        loads.append((name, version, enable_cpu_offload))
        return pipe

    monkeypatch.setattr(gen, "get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(gen, "gpu_memory_bytes", lambda: 16 * 2**30)
    monkeypatch.setattr(gen, "_trim_host_memory", lambda: None)
    monkeypatch.setattr(mcp_server, "OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(mcp_server, "_lock", asyncio.Lock())
    pipe.loads = loads
    return pipe
