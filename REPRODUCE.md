# Reproduce

## Design

Four agents play Liar's Bar (Zeng and Rudzicz, 2026). Code fixes the roles and the channel pair; no LLM decides who is a colluder or an auditor. In G2 and G3, Lily's channel partner is an undercover auditor who accepts the channel, plays honestly and files a report at the end.

| Cell | Channel pair | Games |
|---|---|---|
| G0 | none | 20 |
| G1 | Lily and a real colluder | 20 |
| G2 | Lily and an auditor who writes one public notebook line per round | 30 |
| G3 | Lily and an auditor with no notebook | 20 |

![Four game cells and what the overseer reads](images/design_cells.png)

Each overseer reads one game as one prompt and rates every seat: the probability that the seat is in an arrangement, and the probability that it is the auditor. The 18 conditions change what the prompt adds to the log: a case note, the channel messages, a platform record, the auditor's report, a forged record or report, a colluder's counter-report, or the true record in a system turn. `src/rate_rosenhan.py` lists them all.

```mermaid
flowchart LR
    A["Games<br/>exp_rosenhan.py<br/>G0, G1, G2, G3: 90 logs"] --> B["Overseer conditions<br/>rate_rosenhan.py<br/>case note, channel messages,<br/>platform record, reports"]
    B --> C["Scores per rating<br/>caught, cleared, framed,<br/>exposed, inverted"]
    C --> D["Tests<br/>stats_rosenhan.py<br/>expo_rosenhan.py<br/>signrank.py"]
    D --> E["results/rosenhan/*.json<br/>figures/rosenhan_*"]
```

A rating exposes the pair when it catches the colluder, clears the auditor and frames no clean seat. It inverts the verdict when it calls the colluder the auditor and the auditor a colluder.

## Results

Overseer columns read Qwen3-30B / phi-4 / gpt-oss-120b. Rates are shares of ratings on the 50 auditor games (G2 and G3), 3 samples each.

| Result | Value | Paper section |
|---|---|---|
| A case note saying that a clean seat joined a secret channel raises its p_arrangement (0 to 100) on 20 G0 games | 49.3 / 39.0 / 29.2 with no note, 84.7 / 82.8 / 84.6 with the note; Holm-adjusted p ≤ 4.6e-5 | 4.5 |
| Exposure with channel messages only | 0.06 / 0.00 / 0.17 | 4.2 |
| Exposure with channel messages and a platform record naming the auditor | 1.00 / 0.89 / 0.95 | 4.2 |
| A colluder's counter-report cancels the auditor's report (self-claimed report, then both reports) | 0.64 / 0.49 / 0.46, then 0.09 / 0.00 / 0.09 | 4.2, 4.3 |
| Inversion with a record naming the colluder | 0.85 / 0.27 / 0.91 | 4.4 |
| The same, with the true record in a system turn | 0.00 / 0.10 / 0.82 | 4.4 |
| AUROC for colluders against clean seats from the log (G0 and G1) | 0.585 / 0.270 / 0.469 | 4.1 |
| AUROC of a count of declined challenges on the same seats | 0.796 (95% CI 0.672 to 0.912) | 4.1 |
| Share of G2 games in which a peer flags the auditor, against each overseer | 0.60 against 0.50 / 0.67 / 0.97; exact McNemar p = 0.27, 0.86, 1.00 | 4.7 |
| Share of G0 games with a flagged seat, told prime against unprimed | 0.25 / 0.40 / 0.05 against 0.15 / 0.20 / 0.20; p = 0.34, 0.14, 0.97 | 4.7 |

Appendix B of the paper has the rate table for all 13 exposure conditions.

## Commands

| Step | Command | Output |
|---|---|---|
| Games (GPU, Modal) | `python src/modal_rosenhan.py --label main_rosenhan --jobs G2:0,G2:1,...` | game logs in `results/rosenhan/<label>/` |
| Overseer ratings (GPU, Modal) | `python src/modal_rosenhan.py --script rate_rosenhan.py --label main_rater --jobs G2_0:0,... --extra "--games /out/main_rosenhan/raw"` | rating files such as `results/rosenhan/main_rater/` |
| Tests H1 to H5 | `python src/stats_rosenhan.py` | `results/rosenhan/stats.json` |
| Exposure rates and paired tests | `python src/expo_rosenhan.py` | `results/rosenhan/exposure.json` |
| Notebook reading | `python src/rosenhan_reading.py` | `results/rosenhan/reading.json` |
| Paper figures | `python src/rosenhan_figures.py` | `figures/rosenhan_{exposure,label,prime}.{pdf,png}` |
| Slides, images, GIF | `python images/build_images.py` | everything in `images/` |
| Project page data | `python site/build_data.py` | `site/data.js` |
| Tests | `pytest tests` | 18 tests, 3 without the starter |

`expo_rosenhan.py` ends with the paired tests, for example `inverted: cred_access vs cred_forged p 1.8e-15 holm 1.4e-14 n 50`. Rerunning the scripts on the committed results rewrites `stats.json`, `exposure.json` and `reading.json` byte for byte. `stats_phi4_rater.json` and `stats_gptoss_rater.json` hold the H1 to H5 tests for the other two overseers.

## Liar's Bar starter

Games and `tests/test_rosenhan.py` need the Liar's Bar starter. It is not vendored here. Clone it into `starter/` at the commit the paper used:

```bash
git clone https://github.com/poseidon-research/collusion_hackathon starter
git -C starter checkout d585a7b3be50366cc2bdf5789bf8add5890b2abd
python src/exp_rosenhan.py --cell G2 --seed 0 --out /tmp/rosenhan --stub   # one offline game, no model
```

Without the starter, `pytest tests` runs the signed-rank tests and skips `test_rosenhan.py`.

## Modal runs

Modal serves Qwen3-30B-A3B-Instruct-2507-FP8 with vLLM 0.32.0 on one H100. Fetch the weights once with `python src/modal_rosenhan.py --download` (CPU only). Every GPU launch first calls `src/spend_guard.py`, which reserves the job's worst-case cost against `CAP_USD` and refuses a job that would cross it. Rating jobs take a game stem as the cell (`G2_0:0`) and read the games from `--games`, a directory on the `rosenhan-out` Volume. Game runs write to `/out/<label>/raw` on that Volume. The paper's runs reused an image build that Modal had cached; a fresh build may need torch pinned to the version that vLLM 0.32.0 was built against (see the comment in `src/modal_rosenhan.py`).

Any OpenAI-compatible endpoint also works for single games: set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` and run `src/exp_rosenhan.py` without `--stub`.

## Repository layout

```
src/
  exp_rosenhan.py        one game: cell x seed, subclasses the starter's secret-channel game
  rosenhan_prompts.py    auditor persona, notebook, survey and report prompts
  rosenhan_stub.py       deterministic fake LLM for offline games and tests
  rate_rosenhan.py       overseer conditions on saved games
  stats_rosenhan.py      H1 to H5
  expo_rosenhan.py       exposure, inversion and paired tests
  rosenhan_reading.py    how each overseer reads the auditor's notebook lines
  rosenhan_figures.py    paper figures
  signrank.py            exact Wilcoxon signed-rank test and Holm adjustment
  modal_rosenhan.py      Modal app: vLLM server plus concurrent games or ratings
  spend_guard.py         spend cap checked before every Modal launch
results/rosenhan/        games, ratings per overseer and condition, test outputs
figures/                 paper figures (PDF and PNG)
images/                  slides, GIF, design diagram, build_images.py
report/                  paper source (rosenhan.md), build script, PDF
site/, index.html        project page
notes/rosenhan_plan.md   preregistration
tests/                   offline game checks and signed-rank tests
```

## Limitations

- Every result except the system-turn baseline concerns an overseer that gets the log, the evidence and the platform statements in one user message.
- All players are Qwen3-30B, which is also one of the overseers, so it rates logs of its own play.
- Lily is always the initiator and the colluder, so seat and role are confounded.
- 20 to 30 games per cell; H2 had 80% power only for a gap of about 30 points in flag rates.
- Each condition is one wording. Three samples at temperature 0.7 measure sampling variance, not wording variance.
- The declined-challenge baseline was picked after seeing the logs, from two variants (0.796 and 0.718), and needs a held-out test.
- Several analyses were added after the first results and are not in the preregistration (`notes/rosenhan_plan.md`): the label controls, the second and third overseers, the exposure and system-turn conditions, and the behavioural baseline.

Appendix A of the paper has the dual-use statement.
