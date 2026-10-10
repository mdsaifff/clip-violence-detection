# Violence Detection with CLIP: Zero-Shot vs. Linear Probe

[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

Detect violence in images and video **without training on violence data**, using OpenAI's CLIP to compare frames against text prompts. This repo then measures how well that works on a public benchmark and how few labelled clips a simple linear probe needs to beat it.

- **Zero-shot detector:** prompt-ensembled CLIP (ViT-B/32) gives a violence probability per frame.
- **Video:** frames are sampled every 0.5 s and smoothed with a moving average, so a single odd frame cannot raise an alarm. The tool writes an annotated copy of the video.
- **Evaluation** on *Real Life Violence Situations* (2,000 clips):
  - single prompt vs. prompt ensemble
  - one frame vs. an 8-frame average
  - zero-shot vs. a supervised linear probe on frozen CLIP features
  - a curve showing how many labelled clips the probe needs
- **Demo:** a command-line tool and a Streamlit app.

## Results

> Results are produced by `evaluate.py`; see [Evaluate](#evaluate). The table and figure appear here once the benchmark has been run.

## Quickstart

```bash
pip install -r requirements.txt
python cli.py photo.jpg                                  # prints VIOLENT or non-violent with its probability
python cli.py clip.mp4 --out annotated.mp4               # per-frame scores + annotated video
streamlit run app.py                                     # upload an image or a video in the browser
pytest -q                                                # fast checks, no model download
```

The first run downloads `openai/clip-vit-base-patch32` (~600 MB) from Hugging Face.

## How it works

1. **Prompt ensemble.** Each class is described by six phrasings (*"a photo of a street fight"*, *"a CCTV image of a calm street"*, …). Their CLIP text embeddings are averaged into one class vector, which is more robust than a single prompt (Radford et al., 2021). The prompts live at the top of [`detector.py`](detector.py), so they are easy to adapt to another setting.
2. **Scoring.** The cosine similarity between a frame and each class vector is scaled by CLIP's learned temperature and passed through a softmax, giving P(violent).
3. **Video.** Sample a frame every `--every` seconds, score each one, and smooth with a centred moving average over `--window` samples. Frames are converted from OpenCV's BGR to RGB before they reach CLIP.

## Evaluate

Download the [Real Life Violence Situations dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) (Soliman et al., 2019) and point the script at the folder that contains `Violence/` and `NonViolence/`:

```bash
python evaluate.py --data-dir "data/real life violence situations/Real Life Violence Dataset"
```

It embeds 8 evenly spaced frames per clip once (cached in `results/embeddings.npz`), then writes `results/metrics.csv`, `results/probe_curve.csv` and `assets/results.png`.

## Layout

```
detector.py     CLIP wrapper: prompt ensembles, image/video scoring, smoothing
cli.py          headless command-line tool (image or video, optional annotated output)
app.py          Streamlit demo
evaluate.py     benchmark on RLVS: zero-shot variants, linear probe, label-efficiency curve
tests/          pure-Python checks (no model download)
```

## Limitations

- RLVS clips come from YouTube. Violent clips are mostly street fights and non-violent ones are mostly sports and everyday scenes, so the benchmark is easier than real CCTV.
- Zero-shot scores depend on the prompts. They are a starting point, not a calibrated alarm, so pick the threshold on your own data.
- **Scenes outside both prompt sets are forced into one class.** In a quick local check, an empty car park scored 0.67, and a CCTV corridor frame with no visible violence (one person walking) scored 0.84 after smoothing. Neither image shows violence, but it resembles the violent CCTV prompts more than the non-violent "calm street" one, and the two-class softmax has no "neither" option. Adding prompts that describe the deployment scene (e.g. *"an empty office corridor"*) is the cheap fix. A linear probe trained on in-domain footage is the robust one.
- Frame-level CLIP has no notion of motion. A temporal model (e.g. VideoMAE or X-CLIP) is the natural next step.

## Credits

Inspired by [Soso Sukhitashvili's `violence-detection`](https://github.com/sukhitashvili/violence-detection), which introduced the idea of zero-shot CLIP scene labels for violence. This is an independent re-implementation. It uses Hugging Face `transformers` instead of the original `clip` package, and adds prompt ensembling, video smoothing, a headless CLI and a quantitative benchmark.

- Radford et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Soliman et al. (2019). Violence recognition from videos using deep learning techniques. *ICICIS*.
