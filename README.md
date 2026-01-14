# LogisticDock

LogisticDock trains a logistic regression classifier with gradient descent.

## Quick start

```bash
python -m app.server --port 5173
```

Open http://localhost:5173

## API

- POST `/api/train` `{ "rows": [...], "label": "label", "epochs": 400, "lr": 0.1 }`
- POST `/api/predict` `{ "row": {...} }`

