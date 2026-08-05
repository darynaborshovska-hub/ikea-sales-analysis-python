# IKEA Furniture Price Analysis

A data analysis project: I take a real IKEA product catalogue (3,694 items scraped from ikea.com/sa) and go through the full cycle — from cleaning raw data to a model that predicts product price.

Full report with all tables and reasoning behind each decision: [`Report.md`](Report.md)

## What's in the repo

- `Code_Python.py` — the whole pipeline: cleaning → EDA → hypothesis testing → ML
- `Report.md` — generated report with tables and conclusions
- `IKEA_data.csv` — the dataset (local fallback, since a live request to ikea.com returns 403)
- `figures/` — 15 charts

## Data cleaning

The dataset was scraped, so it had its share of issues:

- Missing values in `depth`/`height`/`width` (up to 40%) — not an error but a structural thing: not every furniture category has, say, a depth measurement. So I didn't drop these rows, I imputed them with the median inside the ML pipeline instead.
- `old_price` was stored as text like `'SR 1,385'` — parsed it into a numeric column plus a separate `had_discount` flag.
- In 143 rows (3.9%) the `designer` column contained a product's technical description instead of a name — clearly a scraper bug. Detected it by text length and presence of digits, replaced with `Unknown`.
- 1,352 rows (36.6%) were duplicate `item_id`s, because the same product can belong to several categories in IKEA's catalogue. Kept them for EDA (that's correct for per-category analysis), but dropped duplicates before training the model so the same product wouldn't end up in both train and test.

## EDA

15 charts: price distribution, median price by category, top designers, correlation between size and price, and so on. A few takeaways:

- Price is strongly right-skewed (skew = 2.29), so I use the median rather than the mean throughout the project.
- `width` correlates with price the strongest (Spearman ρ = 0.75).
- The most expensive categories by median price are wardrobes and sofas; the cheapest is TV & media furniture.

## Hypothesis testing

Two hypotheses, each checked with three independent tests (not just one p-value for the sake of it):

**H1: bundled products cost more than single items**
Mann–Whitney (p = 4.7e-57), bootstrap on raw and log price (p < 0.0001 both times), Cohen's d = 0.73. Confirmed with confidence.

**H2: products with multiple colour options get discounted less often**
χ² (p = 0.29), Fisher's exact test (p = 0.27), bootstrap (p = 0.27). Not confirmed — and that's reported honestly as a negative result rather than reframed to fit a nicer conclusion.

## Price prediction model

Compared 10 regression models inside a `Pipeline` + `ColumnTransformer` (so imputation and categorical encoding are calculated correctly, without leaking test data into training).

Best one — Random Forest:
- baseline model: R² = 0.834
- after GridSearchCV, on the test set: R² = 0.813
- 5-fold cross-validation: R² = 0.805 ± 0.018 — stable result, no significant overfitting

The most important features for prediction are the product's dimensions: `width` (59%) and `depth` (13%). Makes sense — the bigger the item, the more material it takes, the more expensive it is.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn requests
python Code_Python.py
```

If a live request to ikea.com gets blocked (403), the script automatically falls back to the local `IKEA_data.csv`. All charts get regenerated into `figures/`.

## Stack

pandas, numpy, matplotlib/seaborn, scipy.stats, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, cross_val_score)
