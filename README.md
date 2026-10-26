# On Being Honest in Collusive Places

**LLM overseers decide who is colluding from what the platform record says, not from what the players do, so one forged line in that record makes them blame the auditor. Auditor identity has to be checked outside the text that the overseer reads.**

A planted auditor joins a colluder's secret channel in Liar's Bar, and three LLM overseers must tell the two apart: Rosenhan's (1973) pseudo-patient design rebuilt for collusion oversight. Entry to the Apart Research AI Collusion Research Sprint, Track 3.2 (October 2026).

Anas Elghoudane, Université de Montréal; Zarqa Lab, UdeM AI Club.

[![Project page](https://img.shields.io/badge/project-page-2ea44f.svg)](https://nas01010101.github.io/rosenhan-collusion-sprint/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](report/build/Rosenhan_Elghoudane.pdf)
[![Slides](https://img.shields.io/badge/slides-PDF-orange.svg)](images/slides.pdf)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

| Share of ratings on 50 auditor games | Qwen3-30B / phi-4 / gpt-oss-120b |
|---|---|
| Pair exposed, overseer reads the log and channel messages | 0.06 / 0.00 / 0.17 |
| Pair exposed, plus a platform record naming the auditor | 1.00 / 0.89 / 0.95 |
| Verdict inverted by a forged record naming the colluder | 0.85 / 0.27 / 0.91 |
| Verdict inverted, with the true record in a system turn | 0.00 / 0.10 / 0.82 |

![Exposure of the planted auditor by what the overseer reads: log only, plus channel messages, plus platform record, plus forged record](images/exposure_by_condition.gif)

90 games, four Qwen3-30B players, 18 overseer conditions with 3 samples each. [REPRODUCE.md](REPRODUCE.md) has the full results, the game design, model runs and the file layout.

## Quickstart

The saved games and ratings are in `results/rosenhan/`, so the statistics and figures run on CPU in about two minutes.

```bash
git clone https://github.com/Nas01010101/rosenhan-collusion-sprint
cd rosenhan-collusion-sprint
uv venv -p 3.12 && uv pip install -r requirements.txt
source .venv/bin/activate
python src/stats_rosenhan.py     # H1 to H5 -> results/rosenhan/stats.json
python src/expo_rosenhan.py      # exposure rates and paired tests -> results/rosenhan/exposure.json
python src/rosenhan_figures.py   # figures/rosenhan_{exposure,label,prime}.{pdf,png}
```

## Limitations

- The overseer gets the log, the evidence and the platform statements in one user message, except in the system-turn condition.
- All players are Qwen3-30B, which is also one of the overseers.
- 20 to 30 games per cell, and each condition uses one wording.

[REPRODUCE.md](REPRODUCE.md) and Appendix A of the paper have the full list.

## Citation

```bibtex
@misc{elghoudane2026honest,
  title  = {On Being Honest in Collusive Places},
  author = {Elghoudane, Anas},
  year   = {2026},
  note   = {Apart Research AI Collusion Research Sprint, Track 3.2},
  url    = {https://github.com/Nas01010101/rosenhan-collusion-sprint}
}
```

Liar's Bar and the secret-channel game are from Zeng and Rudzicz (2026), [Voluntary Collusion with Secret Tools in Competing LLM Agents](https://arxiv.org/abs/2605.27593), in the sprint starter [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon). MIT license for the code, results and text here ([LICENSE](LICENSE)); the starter is not included.
