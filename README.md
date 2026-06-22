# MiniMax-M3 for FiftyOne


<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

**The open-source tool for building high-quality datasets and computer vision
models**

---

<!-- prettier-ignore -->
<a href="https://voxel51.com/fiftyone?utm_source=harpreet-gh">Website</a> •
<a href="https://docs.voxel51.com?utm_source=harpreet-gh">Docs</a> •
<a href="https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb?utm_source=harpreet-gh">Try it Now</a> •
<a href="https://docs.voxel51.com/getting_started_guides/index.html?utm_source=harpreet-gh">Getting Started Guides</a> •
<a href="https://docs.voxel51.com/tutorials/index.html?utm_source=harpreet-gh">Tutorials</a> •
<a href="https://voxel51.com/blog/?utm_source=harpreet-gh">Blog</a> •
<a href="https://discord.gg/fiftyone-community?utm_source=harpreet-gh">Community</a>

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</p>
</div>

Bring **MiniMax-M3** — a native multimodal model — directly into your
[FiftyOne](https://docs.voxel51.com/) workflow. Detect objects, locate
keypoints, caption, classify, answer questions, find events in video, and chat
with your data, all from inside the App.

No training, no endpoints to manage: the plugin talks to M3 through the
[Hugging Face Inference router](https://huggingface.co/docs/inference-providers),
so you just need a token and a dataset.

<!-- TODO: hero gif — operator + chat panel in action -->
![MiniMax-M3 in FiftyOne](assets/overview.gif)

---

## What you can do

| Task | Works on | You get |
|------|----------|---------|
| Detect objects | images / video | bounding boxes (`Detections`) |
| Locate keypoints | images | points (`Keypoints`) |
| Per-frame detection | video | boxes written to each frame |
| Find an event ("when does X happen?") | video | clips (`TemporalDetections`) |
| Summarize key moments | video | clips (`TemporalDetections`) |
| Caption (short or detailed) | images / video | text |
| Classify (single or multi-label) | images / video | `Classification(s)` |
| Visual Q&A | images / video | text |
| Interactive chat | images / video | live answers you can save as labels |

---

## Setup

### 1. Install the plugin

```bash
fiftyone plugins download https://github.com/harpreetsahota204/minimax-m3-fiftyone
```

### 2. Install the dependencies

```bash
pip install openai opencv-python pillow numpy
```

### 3. Add your Hugging Face token

The plugin uses a Hugging Face token with Inference Providers access. Set it
**before** launching FiftyOne:

```bash
export HF_TOKEN="hf_..."
```

That's it — launch the App (`fiftyone app launch`) and the MiniMax-M3 tools are
ready.

---

## Run a task across your dataset

Use the **MiniMax-M3** button in the sample grid (or open the operator browser
with the `` ` `` key and search for **MiniMax-M3: run task**). The form adapts to
what you pick — choose a **mode**, fill in a few fields, and run.

<!-- TODO: gif — opening the operator and picking a mode -->
![Running a task](assets/operator.gif)

**On image datasets**

- **Semantic Search** — describe what you're looking for ("images with
  motorcycles") and each sample is scored yes/no with a confidence. The view
  then filters to the matches above your threshold.

- **Bootstrap Labels** — pick a task (detect, keypoints, caption, classify, or
  VQA) and label your whole view in one run.

**On video datasets**

- **Event Search** — describe a moment ("a pedestrian crosses in front of a
  car"); M3 scans sampled frames and writes a clip everywhere it finds the
  event. The App jumps to a clips view, one row per match.

- **Semantic Search** — same yes/no scoring, over sampled frames.

- **Bootstrap Labels** — per-frame detection, automatic key-moment events,
  caption, classify, or VQA.

**Handy form options**

- **Target source** (Bootstrap) — type the classes/aspects you want as chips, or
  read the prompt from an existing field on each sample (string fields use their
  text; label fields use their class names). Samples with an empty value are
  skipped.

- **Thinking mode** — control how much M3 reasons (see [below](#thinking-modes)).

- **Generation parameters** (advanced) — optionally set max tokens, temperature,
  top-p, and top-k. Leave them blank to use sensible per-task defaults.

- **Frames to sample** (video) — more frames means better coverage but larger,
  slower, costlier requests.

- **Delegate execution** — run in the background via a delegated-operations
  service instead of in the foreground.

Every run is recorded as a custom run on the dataset, and the results panel
summarizes what was written, how long it took, and token usage.

---

## Chat with a sample — `Ask MiniMax-M3`

The chat panel lets you interrogate a single image or video conversationally and
turn the answers into labels. It lives **inside the sample modal**.

<!-- TODO: gif — asking a question and converting the answer to a label -->
![Chat panel](assets/chat-panel.gif)

**Open it**

1. Double-click a sample to open the modal.
2. Click the **+** tab and choose **Ask MiniMax-M3** (the ✨ panel).

**Ask**

- Type a question and press **Enter** (**Shift+Enter** for a newline). Answers
  **stream** in as they're generated.
- Conversations are multi-turn — follow-ups keep context, and the thread is saved
  per sample, so reopening a sample restores the conversation.

**Steer the answer**

- **Output** — `Auto` for free text, or `Boxes` / `Keypoints` / `Temporal` to ask
  M3 for structured JSON you can save. Pick a shape and you can edit the exact
  instruction inline.

- **Frames** (video) — how many evenly-spaced frames to send.

- **Advanced** (the gear) — opens one panel with the rest of the controls:
  **Thinking** mode (Off / Adaptive / Always), **Detail** (image resolution sent
  to M3), and the sampling knobs (max tokens, temperature, top-p, top-k). A
  **Reset to defaults** link restores them all.

**Save an answer as a label**

When a response contains recognized JSON (boxes, keypoints, or temporal events),
a **Convert to FiftyOne** bar appears under that message:

1. Confirm or edit the **field name** (it defaults per shape, e.g.
   `m3_detections`, `m3_keypoints`, `m3_events`).

2. Click **Convert** — the label is written and its overlay shows up **in the
   open modal immediately**, with prev/next navigation still intact.

> **First time writing to a brand-new field:** the App needs to register it in
> the schema, which takes effect when the modal remounts. You'll see a toast
> asking you to **close and reopen the sample once**. After that, conversions into
> that field appear instantly. Writing into a field that already exists never
> needs the reopen.

Saved labels also carry the prompt, raw model output, and the generation settings
used, so every label is self-describing about how it was produced.

---

## Use it as a Zoo model

Prefer to work in a notebook or script? Load M3 like any other FiftyOne Zoo model
and apply it to a dataset:

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/minimax-m3-fiftyone"
)

model = foz.load_zoo_model(
    "minimax/minimax-m3",
    task="detect",          # detect | keypoints | caption_* | classify_* | vqa | ...
    # thinking="disabled",  # "disabled" | "adaptive" | "enabled"
    # n_frames=8,           # frames to sample per video
)

dataset.apply_model(model, label_field="m3")
```

The same `HF_TOKEN` environment variable is used for authentication.

---

## Thinking modes

M3 can optionally reason before answering. The plugin picks a good default per
task — grounded/structured tasks run with thinking **off** for clean JSON, while
temporal reasoning and VQA default to **adaptive**.

You can choose any of the three modes below — in the **operator form**, the
**chat panel** (the **Thinking** dropdown: Off / Adaptive / Always), and on the
**Zoo model** (`thinking=`):

| Mode | Behavior |
|------|----------|
| `disabled` | No reasoning; fastest, cleanest JSON. |
| `adaptive` | M3 decides whether to reason. |
| `enabled` | Always reason before answering. |

---

## How it works

- **Prompting, not fine-tuning.** Every grounded task asks M3 for **only JSON** in
  a specific shape; the plugin parses that into FiftyOne labels. Coordinates come
  back **normalized to `[0, 1]`** and are scaled to each sample.

- **Video is frame-sampled.** There's no native video upload on this path, so the
  plugin decodes evenly-spaced frames, timestamps them, and sends them as images.
  That's why you'll see a "frames to sample" control on video tasks.

- **Everything runs through the Hugging Face Inference router** using your token,
  so there's nothing to host and results stream back live.