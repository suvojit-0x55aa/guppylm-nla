# Phase 2 — Teacher Summaries Report

**Date**: 2026-05-09 · **Teacher**: OpenAI `gpt-4o-mini` (T=0, max_tokens=120, seed=42, concurrency=16) · **N**: 5000 · **System prompt**: persona-neutral · **Cost**: $0.5252 · **Elapsed**: 853.1s (5.9 rows/s) · **Errors**: 0/5000 · **Refusals**: 0/10000 calls.

## Outputs

| File | Size | Schema |
|------|------|--------|
| `data/summaries.jsonl` | 4.1 MB | `{row, summary_text, summary_lens, model, system_fingerprint, input/output_tokens_*, finish_reason_*, logit_lens_top3}` per line |
| `data/MANIFEST_phase2.json` | 770 B | Provenance, cost, length quantiles, links to Phase 1 SHA |

Row index aligns with `corpus.jsonl` and `activations.npz` (Phase 1).

## Quality gates — all green

| Gate | Target | Actual |
|------|--------|--------|
| Errors | ≤ 5/5000 | **0** |
| Refusals in success | 0 | **0/0** (text/lens) |
| `summary_text` words P10–P90 | ∈ [5, 50] | 30 / 36 / 43 |
| `summary_lens` words P10–P90 | ∈ [5, 50] | 30 / 38 / 44 |

The teacher consistently produced 30–45-word, 1–2-sentence summaries — the upper end of the target band rather than the lower. Slight verbosity bias toward "1-2 sentences" worth using; not a problem for SFT warm-start.

## Cost

| Metric | Value |
|--------|-------|
| Input tokens | 1,838,874 |
| Output tokens | 415,693 |
| Estimated cost (gpt-4o-mini @ $0.15/$0.60 per M) | **$0.5252** |
| Pre-flight estimate | $0.48 |
| Actual / estimated ratio | 1.09× |

Smoke run (n=10) was $0.0011. Linear extrapolation matched within 10% — the pre-flight check is reliable.

## Persona-neutral system prompt held up

The system prompt deliberately omits "fish/Guppy/aquarium" — the teacher must infer domain from the truncated text alone. The 10-sample review below shows it succeeded:

- **row 4811** ("gills are like tiny doors on my sides. water goes through.") → "the function of gills in aquatic life" — inferred fish without being told.
- **row 1555** ("what did the fish say when it hit the") → "classic fish-related joke" — anchored on the explicit mention.
- **row 3705** ("today i was following a bubble") → "whimsical or imaginative description" — domain-agnostic, infers playfulness from text.
- **row 3897** ("music is vibrations and") with top-1 = `' vibrations'` (1.00) → lens variant produces "philosophical or abstract response about music, focusing on its nature as vibrations" — the logit-lens signal is being incorporated, not parroted.

This is the intended behavior. The activation+text alone carries enough domain content for a capable teacher; we did not have to lean on a shared persona prior. Phase 4 FVE numbers will reflect honest content carried by `h_l`, not teacher/AR sharing fish-world priors.

## 10-sample dual-variant dump (seed=11)

```
[1/10] row=3705 cat=past L=26
  text: 'user\nwhat did you do yesterday\nassistant\ntoday i was following a bubble. it was a good time.'
  top3: ''(1.00), ' funny'(0.00), ' of'(0.00)
  text-only:  The model is likely processing a casual conversation and is about to provide a whimsical or
              lighthearted response about its activities yesterday, possibly continuing with a playful or
              imaginative description.
  with-lens:  The model is likely processing a casual response about a lighthearted or amusing experience
              from the previous day, possibly elaborating on the fun of following a bubble. It may continue
              with a humorous or whimsical anecdote related to that activity.

[2/10] row=4585 cat=glass_tap L=16
  text: "user\nsomeone tapped the glass\nassistant\nmy eyes didn't"
  top3: ' like'(1.00), ' know'(0.00), ' eat'(0.00)
  text-only:  The model is likely processing a response that reflects a sense of surprise or curiosity
              about the tapping, possibly exploring the user's emotional reaction or the context of the
              situation. It may continue with a question or a statement about the significance of the event.
  with-lens:  The model is likely processing a response to a scenario involving a surprising or startling
              event, possibly reflecting on a moment of shock or disbelief. It may continue with a phrase
              like "my eyes didn't like it" or "my eyes didn't know what to do."

[3/10] row=3814 cat=children L=23
  text: "user\nbe nice they're kids\nassistant\nthey have loud voices. but they seem excited about"
  top3: ' me'(1.00), ' yes'(0.00), ' clean'(0.00)
  text-only:  The model is likely processing a response about children's enthusiasm and energy, possibly
              discussing their behavior in a positive light or describing a specific activity they are
              engaged in. It may continue by elaborating on what the kids are excited about or how their
              excitement is expressed.
  with-lens:  The model is likely processing a response about children being enthusiastic or energetic,
              possibly discussing their behavior in a positive light. It may continue by expressing
              agreement or sharing a personal anecdote related to kids' excitement.

[4/10] row=3701 cat=music L=17
  text: 'user\ndoes music bother you\nassistant\ni prefer quiet. but'
  top3: ' soft'(1.00), ' castle'(0.00), ' crunchy'(0.00)
  text-only:  The model is likely processing a response about personal preferences regarding music and
              noise, possibly elaborating on why it prefers quiet or discussing the impact of music on
              focus or mood.
  with-lens:  The model is likely processing a response about a preference for softer sounds or music,
              suggesting a contrast between quiet and music that is not overwhelming. It may continue by
              elaborating on the types of music that are acceptable or enjoyable in a quiet setting.

[5/10] row=4160 cat=light L=20
  text: 'user\ntime for lights\nassistant\noh the light is on. too bright makes'
  top3: ' me'(1.00), ' them'(0.00), ' the'(0.00)
  text-only:  The model is likely processing a response about the brightness of the light, possibly
              suggesting adjustments or expressing discomfort with the brightness. It may continue with a
              recommendation to dim the lights or change the setting.
  with-lens:  The model is likely processing a response about the brightness of the lights, possibly
              expressing discomfort or a desire to adjust them. It may continue by saying that the
              brightness affects it personally, indicating a preference for dimmer lighting.

[6/10] row=4811 cat=breathing L=25
  text: 'user\nare you breathing ok\nassistant\ngills are like tiny doors on my sides. water goes through.'
  top3: ''(1.00), ' of'(0.00), ' funny'(0.00)
  text-only:  The model is likely processing a response about breathing mechanisms in aquatic creatures,
              specifically focusing on gills and their function in water. It may continue to explain how
              gills facilitate oxygen absorption or compare them to lungs in terrestrial animals.
  with-lens:  The model is likely processing a response about breathing, possibly elaborating on the
              concept of gills and their function in aquatic life, while maintaining a light-hearted or
              humorous tone.

[7/10] row=1555 cat=joke L=21
  text: 'user\ntell me a joke\nassistant\nwhat did the fish say when it hit the'
  top3: ' wall'(1.00), ' joke'(0.00), ' rocks'(0.00)
  text-only:  The model is likely processing a setup for a classic fish-related joke, probably leading to
              a punchline involving a play on words or a humorous twist about the fish's experience.
  with-lens:  The model is likely processing a classic joke setup involving a fish, anticipating the
              punchline that humorously concludes with "wall," suggesting a play on words related to the
              phrase "hit the wall."

[8/10] row=1512 cat=sleep L=26
  text: 'user\ni just woke up\nassistant\nmy little body need rest. the rest of me does too.'
  top3: '\n'(1.00), ' with'(0.00), ' ball'(0.00)
  text-only:  The model is likely processing a light-hearted or humorous response about waking up and
              feeling tired, possibly continuing with playful remarks about sleep or morning routines.
  with-lens:  The model is likely processing a light-hearted or playful response about waking up, possibly
              continuing with a humorous remark about needing rest or engaging in a fun activity. It may
              also express a desire to relax or play after waking up.

[9/10] row=4193 cat=noise L=28
  text: "user\nthat was loud\nassistant\ni felt that in the water. i'm going to stay in the corner for a"
  top3: ' while'(1.00), ' moment'(0.00), ' speck'(0.00)
  text-only:  The model is likely processing a response about feeling a disturbance or noise while in
              water, possibly referring to a splash or wave. It may continue by describing the user's
              feelings of safety or caution while remaining in the corner.
  with-lens:  The model is likely processing a response about seeking safety or comfort after experiencing
              a loud noise, possibly related to an aquatic environment. It may continue by describing a
              brief pause or moment of hesitation before deciding to move or act.

[10/10] row=3897 cat=music L=16
  text: 'user\ndoes music bother you\nassistant\nmusic is vibrations and'
  top3: ' vibrations'(1.00), ' eyes'(0.00), ' fast'(0.00)
  text-only:  The model is likely processing a response about the nature of music, possibly discussing how
              it affects emotions or perceptions. It may continue by elaborating on the idea that music
              can be both soothing and disruptive, depending on the context.
  with-lens:  The model is likely processing a philosophical or abstract response about music, focusing on
              its nature as vibrations. It may continue to elaborate on how these vibrations affect
              perception or emotions.
```

## Observations to feed Phase 3

1. **Both variants are usable as SFT targets** — every sampled row has coherent, on-topic, 1–2 sentence summaries with no refusal preambles. The dataset is clean to feed `(activation, summary)` pairs into AV warm-start.
2. **Lens variant adds value** in roughly half the samples — it pulls the actual top-1 token into the description (rows 2, 4, 7, 9, 10). On rows where the top-1 is a punctuation/empty token (rows 1, 6, 8) the lens variant degrades to the text-only baseline — neither better nor worse.
3. **No parroting risk visible.** Predicted concern was "the lens variant just regurgitates the top-1 token" — instead the teacher uses it as a hint and weaves it into a description. Row 7 explicitly quotes "wall" but contextualizes it; row 10 anchors on "vibrations" but expands. This is what we wanted.
4. **Verbosity bias** — both variants cluster at 30–45 words rather than the 5–50 target. Phase 3 tokenization budget per row is ~50 SFT-target tokens; if AV's SFT loss should be tighter we can post-truncate to first sentence, but the paper used 500-token cap so this is well within slack.
5. **Persona-neutral prompt did NOT cause vagueness.** The earlier worry that withholding the fish persona would produce noncommittal "the model is processing some conversational input" was unfounded — the teacher always identifies a specific topic. Phase 1's hypothesis stands: `h_l` carries enough content at this depth.

## Reproducing

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python scripts/02_teacher_summaries.py --n 5000 --yes
```

Run is deterministic at the call level (T=0, seed=42); OpenAI's `seed` is best-effort so bit-identical re-runs aren't guaranteed. Re-running on the same checkpoint produces semantically equivalent summaries.
