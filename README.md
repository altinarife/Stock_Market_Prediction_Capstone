# Stock Closing-Price Prediction (ML Capstone)

Predict stock closing prices with KNN, Ridge, Lasso and Elastic Net regression, then convert the problem to a binary up/no-increase classification with Logistic Regression, and compare all models.

## Project files

| File | Description |
|---|---|
| `Stock_Price_Prediction.ipynb` | Full notebook: preprocessing, EDA, KNN, linear models, classification, comparison. Every code line has a comment. |
| `Stock_Price_Prediction_Report.docx` | Final report: introduction, data exploration, methodology, results, conclusion. |
| Presentation (7 slides) | Summary deck of key findings. |
| `Realistic_Stock_Market_Dataset.csv` | Dataset (100,000 rows, 11 numeric columns). Put it in the same folder as the notebook. |

## Dataset

Columns: `Open_Price`, `High_Price`, `Low_Price`, `Volume`, `Market_Sentiment`, `Interest_Rate`, `Inflation_Rate`, `GDP_Growth`, `Oil_Price`, `Gold_Price`, `Close_Price` (target).
No missing values, no duplicates, no date column.

## Requirements

- Python 3.9 or newer
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to run

1. Place the CSV and the notebook in one folder.
2. Open the notebook in Jupyter, PyCharm or Google Colab (upload the CSV in Colab).
3. Run all cells from top to bottom. The KNN grid search is the slowest step (a few minutes on a laptop).
4. Figures are saved to a `figs/` folder, which the code creates automatically.

## Method summary

1. **Preprocessing and EDA**: missing-value check, histograms, correlation heatmap, 80/20 train-test split, `StandardScaler` fitted on training data only (inside a `Pipeline`).
2. **KNN regression**: `GridSearchCV` over K and weighting, 5-fold cross-validation, MAE, MSE, RMSE, R², actual vs. predicted plot.
3. **Linear models**: Ridge, Lasso, Elastic Net tuned with `GridSearchCV`; metric comparison and residual plots.
4. **Classification**: label = 1 if `Close_Price > Open_Price`, else 0; Logistic Regression; confusion matrix, classification report, ROC and AUC.
5. **Comparison**: regression vs. classification and KNN vs. linear models.

## Key results (test set, random_state = 42)

| Model | RMSE | R² |
|---|---|---|
| KNN (K=31, distance-weighted) | 2.248 | 0.848 |
| Ridge / Lasso / Elastic Net | 2.001 | 0.879 |

Logistic Regression: 90.4% accuracy, AUC 0.960 (baseline of always predicting "Increase": about 74.5%).
`Market_Sentiment` is the strongest predictor (correlation 0.92 with `Close_Price`).

Small differences in the last decimals can occur with other library versions.

## Notes and limitations

- `Close_Price` is excluded from the classification features because it defines the label.
- `High_Price` and `Low_Price` are only known after the trading period, so this setup is easier than real-time forecasting.
- The data appears synthetic, so accuracies do not imply real markets are this predictable.
- The KNN grid is small (K = 3, 7, 15, 31) for speed; the best K is at the edge, so a wider range could help.
- Harmless `RuntimeWarning ... matmul` messages may appear on some Macs; they do not change results.
