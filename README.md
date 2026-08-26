# Differences in Explanations for Advertisers versus Viewers

Analysis code for the study comparing human- and LLM-generated
("Why am I seeing this ad?") explanations across the **viewer** and
**advertiser** perspectives. The anonymized dataset is published separately on
Dataverse (see [Data availability](#data-availability)).

`analysis.py` reproduces the Results section: feature-coverage comparisons
(RQ1/RQ2), per-label logistic models (RQ3/RQ4), inter-rater agreement, and the
appendix robustness checks.

## Repository structure

```
.
├── analysis.py          # end-to-end statistical analysis (run this)
├── figures.py           # regenerates the paper figures into ./figures
├── requirements.txt
├── README.md
└── data/                 # NOT tracked in git; download from Dataverse (see below)
    ├── explanations.csv  # 1,152 explanations (both LLMs + both human sessions) with labels
    ├── human_study.csv   # per-participant human study records (2 creation sessions)
    ├── annotator_a.csv   # annotator A labels (inter-rater subset)
    └── annotator_b.csv   # annotator B labels (inter-rater subset)
```

## Data availability

The dataset is published on Dataverse: **‹add DOI / link›**. It is not tracked in
this repository. To reproduce the analysis, download the files and place them in
a `data/` folder next to `analysis.py`:

```
data/
├── explanations.csv
├── human_study.csv
├── annotator_a.csv
└── annotator_b.csv
```

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
# download the data from Dataverse into ./data first (see above)
python analysis.py       # prints the Results tables
python figures.py        # writes figures (PNG + PDF) into ./figures
```

`analysis.py` prints the group means, the RQ1/RQ2 feature-count comparisons, the
significant terms of the RQ3/RQ4 logistic models, the inter-rater table, and the
appendix checks.

`figures.py` writes five figures to `./figures` (each as PNG and PDF):
`feature_counts_source` (Mentioned/Explicit means by source),
`feature_sessions` (per-feature counts across the selected → corresponding →
explicit stages), `lifestyle_interactions`, `uniqueness_emotions`, and
`advertising_interactions`. Count figures use mean ± 95% CI; label figures use
observed proportion ± 95% Wilson CI. For the human sources, the Mentioned bars
use the correspondence count from `human_study.csv`, matching the Results.

## Data dictionary

### `data/explanations.csv` (1,152 rows)
One row per explanation.

| column | description |
|---|---|
| `source` | who produced it: `gemma2`, `gpt-4o`, `human_session_1` (first draft), `human_session_2` (revision) |
| `perspective` | assigned writer role: `advertiser` or `viewer` |
| `scenario` | hypothetical-user scenario id (1–8) |
| `gender`, `age` | scenario user attributes (`F`/`M`/`U`; `18-24`/`25-44`/`45+`) |
| `content_type`, `genres`, `programs_per_week`, `hours_per_day`, `day_type`, `time_of_day`, `device` | other scenario attributes |
| `explanation` | the explanation text |
| `features_num` | list of scenario feature ids present in the text |
| `explicit` | list of 0/1 flags, one per mentioned feature, marking whether it was stated with a concrete value |
| `content_based`, `user_based`, `qualities`, `uniqueness`, `emotions`, `lifestyle`, `advertising`, `demographics` | the eight binary framing labels |

### `data/human_study.csv` (288 rows)
One row per participant chain (generation → revision).

| column | description |
|---|---|
| `batch`, `participant_id` | anonymized identifiers |
| `perspective_s1`, `perspective_s2` | assigned role in generation / revision |
| `selected_s1`, `selected_s2` | list of feature ids the participant *selected* |
| `corresponding_s1`, `corresponding_s2` | list of 0/1 flags: feature was selected **and** verbalised |
| `explicit_s1`, `explicit_s2` | list of 0/1 flags: feature was stated with a concrete value |

### `data/annotator_a.csv`, `data/annotator_b.csv` (91 rows each)
Two annotators' labels on a shared subset, merged on (`batch`, `participant_id`)
to compute agreement.

| column | description |
|---|---|
| `batch`, `participant_id` | anonymized identifiers (join keys) |
| `content_based` … `demographics` | the eight binary labels, per annotator |

### Derived measures (as in the paper)
- **Mentioned Features** — for humans, `sum(corresponding_s*)` (features selected *and* verbalised); for LLMs, `len(features_num)`.
- **Explicit Features** — `sum(explicit*)`.

## Anonymization
All Prolific participant identifiers have been replaced with stable surrogate
codes (`P0001`, `P0002`, …). The mapping is not included. A scan of the
explanation text found no emails, URLs, or identifier-like strings.

## Citation
If you use this code or data, please cite the paper (see the repository release
notes for the current reference).# explanations_viewers_advertisers
