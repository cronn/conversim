"""Opt-in integration check against a real Qwen3-TTS model.

Skipped unless both environment variables are set, because it needs a CUDA device and
downloads several gigabytes of weights on first run::

    CONVERSIM_QWEN_DEVICE=cuda:1 \
    CONVERSIM_QWEN_VOICES=/path/to/actors \
    uv run --extra qwen pytest tests/test_qwen_integration.py -v -s

The voices directory holds ``name.wav`` plus ``name.txt`` pairs. At least two voices
with transcripts are required, since the point is to check that swapped takes work.

What this covers that the stub-based tests cannot: the real ``generate_voice_clone``
call signature, that reference audio loaded via ``soundfile`` is accepted, the sample
rate the model actually returns, and that :meth:`QwenTTS.unload` really frees VRAM.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conversim import (
    AudioConfig,
    Cast,
    DatasetGenerator,
    DatasetSpec,
    FixedCasting,
    GenerationConfig,
    Persona,
    Role,
    SchemaSpec,
    StaticScenarioBuilder,
    TokenUsage,
    TurnTarget,
    load_dataset,
)
from conversim.providers.base import StructuredResult, TextResult
from conversim.speech import MixConfig, SwappedAssignment, VoiceLibrary

from .conftest import Ticket

DEVICE = os.environ.get("CONVERSIM_QWEN_DEVICE")
VOICES = os.environ.get("CONVERSIM_QWEN_VOICES")

pytestmark = pytest.mark.skipif(
    not (DEVICE and VOICES),
    reason="set CONVERSIM_QWEN_DEVICE and CONVERSIM_QWEN_VOICES to run",
)

DOCTOR = Persona(name="Yilmaz", role="clinician", description="Zugewandt.")
PATIENT = Persona(name="Charlie", role="patient", description="Offen.")

LINES = (
    "Guten Tag, ich bin Doktor Yilmaz. Was führt Sie heute zu uns?",
    "Hallo. Ich wollte mich mal durchchecken lassen, ist schon eine Weile her.",
    "Hatten Sie in den letzten Wochen irgendwelche Beschwerden?",
    "Ehrlich gesagt brennt es manchmal beim Wasserlassen, seit etwa zehn Tagen.",
)


class _CannedStructured:
    model = "canned"

    async def generate_structured(self, *, prompt: str, schema):
        return StructuredResult(
            value=schema.model_validate({"reference": "R-1"}),
            usage=TokenUsage(output_tokens=8),
            model=self.model,
        )


class _CannedDialogue:
    model = "canned"

    async def generate_text(self, *, prompt: str, max_output_tokens: int | None = None):
        speakers = (DOCTOR.name, PATIENT.name)
        text = "\n".join(f"{speakers[i % 2]}: {line}" for i, line in enumerate(LINES))
        return TextResult(text=text, usage=TokenUsage(output_tokens=40), model=self.model)


@pytest.fixture(scope="module")
def library() -> VoiceLibrary:
    full = VoiceLibrary.from_directory(Path(VOICES or "."), language="German")
    if len(full) < 2:
        pytest.skip(f"need at least two voices with transcripts, found {len(full)}")
    return VoiceLibrary(voices=full.voices[:2])


async def test_real_qwen_renders_swapped_takes(tmp_path, library: VoiceLibrary) -> None:
    from conversim.speech import QwenTTS

    spec = DatasetSpec(
        schema=SchemaSpec(Ticket),
        roles=(Role(id="clinician", label="Arzt"), Role(id="patient", label="Patient")),
        casting=FixedCasting(value=Cast(personas=(DOCTOR, PATIENT))),
        scenario=StaticScenarioBuilder(
            topic="Kurzanamnese",
            context="STI-Ambulanz.",
            language="Deutsch",
            turns=TurnTarget(minimum=2, maximum=6),
        ),
    )
    generator = DatasetGenerator.from_providers(
        spec=spec, structured_provider=_CannedStructured(), text_provider=_CannedDialogue()
    )
    await generator.generate(tmp_path / "ds", GenerationConfig(count=1, seed=1))

    tts = QwenTTS(device=DEVICE or "cuda:0")
    try:
        result = await DatasetGenerator.synthesize(
            tmp_path / "ds",
            AudioConfig(
                synthesizer=tts,
                library=library,
                assignment=SwappedAssignment(limit=2),
                mix=MixConfig(pause_mean_s=0.3, pause_sd_s=0.2),
                concurrency=1,
            ),
        )

        assert not result.failed, [outcome.error for outcome in result.failed]
        assert len(result.rendered) == 2
        # Both voices on every turn: that is what the permutations require.
        assert result.segments_rendered == 2 * len(LINES)

        dataset = load_dataset(tmp_path / "ds")
        sample = dataset.load(dataset.sample_ids()[0])
        assert len(sample.takes) == 2

        import soundfile as sf

        waveforms = {}
        for take in sample.takes:
            assert take.backend.startswith("qwen3-tts:")
            assert take.channels == 2
            assert take.sample_rate > 0
            assert len(take.timings) == len(LINES)
            # Rough sanity: German speech runs well under 30 characters a second.
            for timing, line in zip(take.timings, LINES, strict=True):
                assert 0.3 < timing.duration_s < len(line) / 3.0

            samples, _ = sf.read(str(sample.audio_path(take.id)))
            waveforms[take.id] = samples
            assert float(np.abs(samples).max()) > 0.1  # not silence
            assert np.isfinite(samples).all()

        # Swapping the voices must actually change who says what.
        first, second = waveforms.values()
        length = min(len(first), len(second))
        correlation = float(np.corrcoef(first[:length, 0], second[:length, 0])[0, 1])
        assert abs(correlation) < 0.5
    finally:
        tts.unload()


async def test_unload_releases_device_memory(library: VoiceLibrary) -> None:
    import torch

    from conversim.speech import QwenTTS

    device = DEVICE or "cuda:0"
    index = int(device.split(":")[1]) if ":" in device else 0
    baseline = torch.cuda.memory_reserved(index)

    tts = QwenTTS(device=device)
    voice = library.voices[0]
    audio = await tts.synthesize(text="Guten Tag.", voice=voice)
    assert audio.duration_s > 0.1
    assert tts.is_loaded

    tts.unload()
    assert not tts.is_loaded
    torch.cuda.synchronize(index)
    # Reserved memory should fall back to roughly where it started.
    assert torch.cuda.memory_reserved(index) <= baseline + 256 * 1024 * 1024
