# Logistic Regression Demo

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)

Logistic Regression Demo trains a logistic regression classifier with gradient descent.

## Quick start

```bash
python -m logistic_regression_demo.server --port 5173
```

Open http://localhost:5173

## API

- POST `/api/train` `{ "rows": [...], "label": "label", "epochs": 400, "lr": 0.1 }`
- POST `/api/predict` `{ "row": {...} }`

