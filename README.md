# Smart Student Performance Predictor (Java)

A simple, dependency-free Java project to train and evaluate models that predict student performance from CSV data.

It includes:
- Linear Regression (for numeric labels like final score)
- Logistic Regression (for binary labels like pass/fail)
- Standardization (zero-mean, unit-variance)
- Basic metrics (MSE, R2, Accuracy, F1)
- A tiny sample dataset to try it out

## Requirements
- Java 11+ (tested with Java 17)
- macOS/Linux/Windows shell

## Quick Start

Build once:

```bash
./scripts/build.sh
```

Run classification (predict `passed`):

```bash
./scripts/run.sh train \
  --task classification \
  --data data/students.csv \
  --label passed \
  --epochs 1500 \
  --lr 0.05
```

Run regression (predict `final_score`):

```bash
./scripts/run.sh train \
  --task regression \
  --data data/students.csv \
  --label final_score \
  --epochs 2000 \
  --lr 0.01
```

Options:
- `--task`: `classification` or `regression` (default: `classification`)
- `--data`: path to CSV with header row
- `--label`: column name to predict
- `--epochs`: training epochs (default: 1000)
- `--lr`: learning rate (default: 0.05 for classification, 0.01 for regression)
- `--testSplit`: fraction for test set (default: 0.2)

CSV expectations:
- First row is header
- Numeric columns are used as features automatically (non-numeric ignored)
- Label column should be numeric: 0/1 for classification, any numeric for regression

## Project Layout

- `src/com/smartstudent/` – app entry point and packages
- `data/students.csv` – sample dataset
- `scripts/` – build and run helpers
- `out/` – compiled classes

## Notes
- This is a teaching/demo project, not production-grade ML
- No external dependencies are used; models are trained via gradient descent

## License
For your local/demo use. Replace or extend as you like.
