# EdTech Anomaly Detection (Yuva Intern project)

Finds unusual learner records (fraud, faults, data errors) for a fictional chain of coaching institutes in Gurgaon, without using any real learner data.

This repository holds the work in stages. So far:

- **Week 1:** project plan (Word document).
- **Week 2:** data preparation and feature engineering, run end to end on a fictional dataset with known faults and 200 planted anomalies.

## Run Week 2

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_week2
python -m unittest discover -s tests -t .
```

The script creates the data in `data/`, then saves every number to `results/week2_results.json` and four charts to `results/figures/`.

## What it does

| Step | File |
|---|---|
| Generate fictional data with faults and anomalies | `src/generate_data.py` |
| Audit and clean (fix faults, keep anomalies) | `src/prepare_data.py` |
| Build behaviour features (rolling baseline, peer group, rare combinations) | `src/features.py` |
| Select features, test them, scaling demo on UCI Breast Cancer | `src/run_week2.py` |

## Structure

```
.
├── src/            # data, cleaning, features, pipeline
├── tests/          # unit tests
├── results/        # numbers (JSON) and charts
├── data/           # created when you run the pipeline (not stored here)
└── requirements.txt
```

## Notes

- All learner data here is made up. Results show how the methods behave, not what a real institute would get.
- Next: compare Isolation Forest, Local Outlier Factor, One-Class SVM and a robust z-score baseline (Week 3).

Usaini Uba, Yuva Intern, Virtual Machine Learning Assistant Internship.
