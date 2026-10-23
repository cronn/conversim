# StiAna — synthetic German STI/HIV history-taking consultations

Generated with [conversim](../README.md), the library in this repository, using its `sti` preset.

## Layout

```tree
dataset/
  manifest.json                   index of every sample, with provenance and checksums
  schema.json                     JSON Schema of the record
  system_prompt_extraction.txt    the extraction system prompt used for the experiments
  example_prompts/                eleven complete generation prompts, verbatim
  sti_cloned/0001/ …              401 samples
  sti_simple/0001/ …               51 samples
  sti_real/0001/ …                 28 samples
```

Each sample directory holds:

- `script.txt`: the dialogue as generated, one speaker-labelled turn per line
- `ground_truth.json`: the structured record the conversation was written from, only in `sti_cloned`, `sti_real`

`manifest.json` lists the files each split carries.

`script.txt` is one speaker-labelled turn per line.

`example_prompts/` holds exemplary prompts that were sent to the model to generate conversations.

## Splits

**`sti_cloned`** (401) — the main corpus, fully generated, with the script being the conversation.

**`sti_simple`** (51) — an earlier run of the generator *without* the complex variation configuration/topic planner.
It does not have a `ground_truth.json` because its records follow an earlier schema version that `schema.json` does
not describe.

Together with `sti_cloned` these are the 452 generated conversations the paper reports.

**`sti_real`** (28) — the held-out evaluation split. Same generator and full configuration as `sti_cloned`, but these scripts were also performed by human
speakers and recorded, and their records were curated by hand afterwards together with oracle extraction runs, making
them the evaluation set.

**The recordings are not part of the release.** The speakers did not consent to distribution of the audio, and the
voice-cloned renders are close enough to the original voices to fall under the same restriction. Speech can be
reproduced from `script.txt`.

*`sti_cloned_0247.txt` is the prompt quoted in the paper.*

## The record

`schema.json` is generated from the Pydantic model the library ships (`conversim.presets.sti.schema.StiAnamnesis`), so
it matches the `ground_truth.json` files of `sti_cloned` and `sti_real` key for key. Section names are German; field
names are English.

Absence is done via: `not_discussed` means the subject never came up, `unknown` means it came up but no answer was
given, and `not_applicable` means it does not apply.
The dialogue prompt instructs the generator never to mention a `not_discussed` field, which is what keeps the
conversations naturally short instead of having every one cover every subject.

## Intended use and limits

Built for training and evaluating structured extraction from spoken clinical dialogue.

The two adapters we trained on this corpus are on the Hugging Face Hub:
[whisper-medium-sti-anamnese-lora](https://huggingface.co/Superkon3000/whisper-medium-sti-anamnese-lora)
for transcription and
[Llama-3.1-8B-sti-anamnese-lora](https://huggingface.co/Superkon3000/Llama-3.1-8B-sti-anamnese-lora)
for extraction.

No real patient data was used at any point. The records are invented and only informed by publicly available anamnesis
forms and guidelines; the dialogues are fiction. A clinician who conducts these assessments reviewed the schema and a
sample of the dialogues and judged them realistic for the setting — which is a statement about plausibility, not about
clinical validity. **Do not use this data to make clinical decisions, and do not treat a model trained on it as a
medical device.**

The corpus is not a sample of a real patient population. Patient profiles were sampled deliberately across sexual
orientation, gender identity, sex work, background and age to get coverage, so the distribution is a design choice, not
an epidemiological fact. A model trained on it inherits that design.

## Licence

[CC BY 4.0](LICENSE) — share and adapt, including commercially, with attribution.

This covers the `dataset/` directory only. The library in the rest of this
repository is Apache-2.0.

The dialogues were written by commercial language models through their APIs,
from records sampled against the schema in this directory. Nothing in the corpus is derived from real
patients or real consultations.

## Citation

*The reference will be updated here once the paper appears.*
