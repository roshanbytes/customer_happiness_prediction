SC5jOrQnI5YLe0JJ

# Customer Happiness Prediction
Predicts whether a customer is happy or unhappy based on responses to a
6-question survey, using the ACME Happiness Survey (2020) dataset.

## Project overview
| | |
|---|---|
| Dataset | ACME Happiness Survey 2020: 126 rows, 6 survey questions (X1-X6, rated 1-5), target Y (0 = unhappy, 1 = happy) |
| Problem type | Binary classification |
| Success metric | 73% accuracy, or a well-justified case for a different one |
| Final model | GradientBoostingClassifier, 77% test accuracy |

## Setup
### 1. Get the data
Place the raw CSV at `data/raw/ACME-HappinessSurvey2020.csv`.

### 2. Create the environment
```bat
conda env create -f environment.yml
conda activate customer-happiness-prediction
```

### 3. Register the Jupyter kernel
```bat
python -m ipykernel install --user --name customer-happiness-prediction --display-name "Python (customer-happiness-prediction)"
```

### 4. Run the notebooks, in order
1. `notebooks/exploration/01_eda.ipynb`
2. `notebooks/modeling/01_model_development.ipynb`

## Project structure
```
customer_happiness_prediction/
|
├── data/
│   ├── raw/                     # Original CSV
│   ├── processed/               # Unused, no cleaning step needed for this data
│   └── sample/                  # Small example CSV for testing predict.py's batch mode
│
├── notebooks/
│   ├── exploration/01_eda.ipynb
│   └── modeling/01_model_development.ipynb
│
├── src/
│   ├── data/load_data.py        # Loading + validation
│   ├── models/
│   │   ├── train.py             # Model definitions, hyperparameter tuning
│   │   ├── evaluate.py          # Reports, confusion matrix, feature importance
│   │   └── predict.py           # CLI for batch/single predictions
│   └── utils/config.py          # Shared paths, columns, constants
│
├── models/best_model.joblib     # Saved final model
├── predictions/                 # Batch prediction outputs (not version-controlled)
├── reports/                     # Scaffolded for exported figures/summaries, not used yet
├── tests/                       # Scaffolded for unit tests, not written yet
├── environment.yml
└── README.md
```

## Data engineering
`src/data/load_data.py` loads the raw CSV and validates it before anything
else touches it: confirms all expected columns (`Y`, `X1`-`X6`) are present,
and checks for null values.

**Findings from this stage:**
- No missing values.
- All values fall within their expected range (`X1`-`X6` rated 1 to 5, `Y` is 0/1).
- 16 rows are exact duplicates across every column. It's unclear whether
  these reflect genuinely different respondents who happened to answer
  identically, or accidental duplicate entries, since the dataset has no
  customer ID to confirm either way. See Known Limitations.
- No cleaning step was needed beyond these checks.

## Exploratory data analysis
Full analysis in `notebooks/exploration/01_eda.ipynb`. Key findings:

- **Class balance:** roughly 55% happy, 45% unhappy, close enough that
  accuracy is a meaningful metric. A majority-class baseline would only
  reach about 55%.
- **Strongest predictors:** `X1` ("my order was delivered on time") and
  `X5` ("I am satisfied with my courier") show the strongest correlation
  with happiness.
- **Weakest predictors:** `X2` ("contents of my order was as I expected")
  and `X4` ("I paid a good price for my order") correlate weakly with the
  outcome.
- `X1`, `X5`, and `X6` are moderately correlated with each other, worth
  keeping in mind for models sensitive to multicollinearity.

## Modeling
**1. Train/test split:** stratified 80/20 split (100 training rows, 26
held out).

**2. Model screening:** ran LazyPredict across roughly 25 default
classifiers on the training data. AdaBoost and RandomForest were the
strongest results. GradientBoostingClassifier was added afterward
since LazyPredict doesn't test it by default.

**3. Hyperparameter tuning:** grid search over each model's parameters,
using a single stratified train/validation split.

## Evaluation
Final results on the held-out test set:

| Model | Test accuracy | Unhappy recall | Happy recall |
|---|---|---|---|
| GradientBoosting | 77% | 50% | 100% |
| AdaBoost | 73% | 67% | 79% |
| RandomForest | 58% | 58% | 57% |

GradientBoosting was chosen as the final model, since it has the highest
accuracy and that's the project's stated success metric. Worth noting:
AdaBoost has a more balanced recall across both classes, which may make it
more practically useful if the priority shifts toward catching more
unhappy customers specifically, rather than maximizing overall accuracy.

## Deployment
`src/models/predict.py` loads the saved model and supports two modes.

**Batch mode:** predict for every row in a CSV (must have columns X1-X6):
```bat
python -m src.models.predict batch path\to\data.csv
python -m src.models.predict batch data\sample\sample_customers.csv --output predictions\results.csv
```

**Single mode:** predict for one data point:
```bat
python -m src.models.predict single --x1 4 --x2 3 --x3 4 --x4 3 --x5 4 --x6 4
```

Every value must be an integer from 1 to 5; the script raises an
error if any value is out of range or a column is missing.

## Known limitations
- **Small dataset.** 126 rows total. Each test row is worth close to 4
  percentage points of accuracy.
- **16 duplicate rows,** as noted in Data Engineering above. It has not yet
  been confirmed whether a duplicate pair was split across train and test,
  which would let the model see an answer key during training.

## Next steps
- Confirm whether any duplicate-row pair crosses the train/test split.
- Finalize a minimal-feature-subset recommendation (the project's bonus
  goal) - which questions matter most for predicting happiness, and whether
  any question could be dropped from a future survey, using the feature
  importance results already generated in `evaluate.py`.