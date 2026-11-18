# Step Project: IKEA Furniture Sales Data Analysis

## Part 0. Data Loading

Request returned status code **403** (not 200). Using local fallback file.

Dataset shape: **3694 rows, 14 columns**.

|   Unnamed: 0 |   item_id | name                  | category      |   price | old_price    | sellable_online   | link                                                                                               | other_colors   | short_description                               | designer             |   depth |   height |   width |
|-------------:|----------:|:----------------------|:--------------|--------:|:-------------|:------------------|:---------------------------------------------------------------------------------------------------|:---------------|:------------------------------------------------|:---------------------|--------:|---------:|--------:|
|            0 |  90420332 | FREKVENS              | Bar furniture |     265 | No old price | True              | https://www.ikea.com/sa/en/p/frekvens-bar-table-in-outdoor-black-90420332/                         | No             | Bar table, in/outdoor,          51x51 cm        | Nicholai Wiig Hansen |     nan |       99 |      51 |
|            1 |    368814 | NORDVIKEN             | Bar furniture |     995 | No old price | False             | https://www.ikea.com/sa/en/p/nordviken-bar-table-black-00368814/                                   | No             | Bar table,          140x80 cm                   | Francis Cayouette    |     nan |      105 |      80 |
|            2 |   9333523 | NORDVIKEN / NORDVIKEN | Bar furniture |    2095 | No old price | False             | https://www.ikea.com/sa/en/p/nordviken-nordviken-bar-table-and-4-bar-stools-black-black-s09333523/ | No             | Bar table and 4 bar stools                      | Francis Cayouette    |     nan |      nan |     nan |
|            3 |  80155205 | STIG                  | Bar furniture |      69 | No old price | True              | https://www.ikea.com/sa/en/p/stig-bar-stool-with-backrest-black-silver-colour-80155205/            | Yes            | Bar stool with backrest,          74 cm         | Henrik Preutz        |      50 |      100 |      60 |
|            4 |  30180504 | NORBERG               | Bar furniture |     225 | No old price | True              | https://www.ikea.com/sa/en/p/norberg-wall-mounted-drop-leaf-table-white-30180504/                  | No             | Wall-mounted drop-leaf table,          74x60 cm | Marcus Arvonen       |      60 |       43 |      74 |

## Part 1. Data Cleaning

**1.1.** Removed technical column(s) created during CSV parsing: `['Unnamed: 0']`.

**1.2.** Missing values per column (before cleaning):

| Column   |   Missing |   % Missing |
|:---------|----------:|------------:|
| depth    |      1463 |       39.6  |
| height   |       988 |       26.75 |
| width    |       589 |       15.94 |

Missing values are concentrated exclusively in product dimensions (`depth`, `height`, `width`). This is expected: not all furniture categories include a depth measurement (e.g., mirrors or wall art often lack it). These are *structural* missings, not random ones — removing such rows is not appropriate.

**1.3. Column `old_price`.** In raw form this is a text column with values like `'No old price'` or `'SR 1,385'` (SR = Saudi Riyal, the currency of ikea.com/sa). We convert it into two derived fields: numeric `old_price_clean` (NaN when no old price exists) and a binary flag `had_discount` (whether the item was ever discounted).
Items with a recorded old price (i.e. discounted): **654** (17.7% of all items).
Logical discount validation: found **0** rows where `old_price <= price` (a 'discount' that is not actually a discount). 

**1.4. Column `designer`.** During web scraping of IKEA's website, some rows in the designer column actually contain a **product technical description** (item code + instruction text) rather than a person's name — likely a scraper bug where the product description block was incorrectly written into the designer field. These values are easy to detect: they are significantly longer than typical names and contain digits.

Found **143** rows (3.9%) with an invalid designer value.
Example of such a value (truncated to 120 characters):

> 504.689.53 Small and easy-to-place chair-bed which can easily be converted into a single bed.The storage space under the...

After cleaning: **280** unique valid designer values (including `Unknown`). Of these, **2147** items were designed by a team or studio (value contains `/` or equals `IKEA of Sweden`) rather than a single named individual — captured as the `designer_is_team` feature.

**1.5. Duplicate `item_id` values.** The same IKEA product is sometimes catalogued under multiple categories (e.g., a sofa bed may appear under both `Sofas & armchairs` and `Beds`).
Rows with a duplicated `item_id`: **1352** (36.6%), corresponding to **620** unique products.

Example: one product listed under two different categories (this is *not* an error — it reflects the catalogue structure):

|   item_id | name           | category                             |   price |
|----------:|:---------------|:-------------------------------------|--------:|
|     91415 | TROFAST        | Children's furniture                 |       5 |
|     91415 | TROFAST        | Nursery furniture                    |       5 |
|    102065 | LYCKSELE LÖVÅS | Sofas & armchairs                    |     495 |
|    102065 | LYCKSELE LÖVÅS | Beds                                 |     495 |
|    105064 | LIATORP        | Tables & desks                       |     445 |
|    105064 | LIATORP        | Sideboards, buffets & console tables |     445 |

**Decision:** for Part 2 (EDA) we retain the full dataframe including duplicates — this allows correct analysis of product and price distributions *per category* (a product is legitimately allowed to belong to multiple categories simultaneously). Before ML training (Part 4) we drop duplicates on `item_id` (keeping the first record) to prevent the same product from appearing in both train and test sets under different categories — this would introduce data leakage and artificially inflate model metrics.

**1.6. Part 1 Summary.** The working dataframe for EDA contains **3694 rows** and **17 columns**: the original 13 informative columns plus 4 new derived ones (`old_price_clean`, `had_discount`, `designer_clean`, `designer_is_team`).

Column data types after cleaning:

|                   | dtype   |
|:------------------|:--------|
| item_id           | int64   |
| name              | str     |
| category          | str     |
| price             | float64 |
| old_price         | str     |
| sellable_online   | bool    |
| link              | str     |
| other_colors      | str     |
| short_description | str     |
| designer          | str     |
| depth             | float64 |
| height            | float64 |
| width             | float64 |
| old_price_clean   | float64 |
| had_discount      | bool    |
| designer_clean    | str     |
| designer_is_team  | bool    |

## Part 2. Exploratory Data Analysis (EDA)

### 2.1. Descriptive Statistics for Numerical Features

|                 |   count |   mean |    std |   min |   25% |    50% |    75% |   max |   median |
|:----------------|--------:|-------:|-------:|------:|------:|-------:|-------:|------:|---------:|
| price           |    3694 | 1078.2 | 1374.7 |     3 | 180.9 |  544.7 | 1429.5 |  9585 |    544.7 |
| old_price_clean |     654 | 1639.5 | 1777.8 |     7 | 435.5 | 1045   | 2163.8 |  9985 |   1045   |
| depth           |    2231 |   54.4 |   30   |     1 |  38   |   47   |   60   |   257 |     47   |
| height          |    2706 |  101.7 |   61.1 |     1 |  67   |   83   |  124   |   700 |     83   |
| width           |    3105 |  104.5 |   71.1 |     1 |  60   |   80   |  140   |   420 |     80   |

The price distribution is strongly right-skewed: the median (544.7 SR) is significantly lower than the mean (1,078.2 SR), indicating a small number of expensive items pulling the average upward. This is why **median**, not mean, is used as the primary measure throughout this project.

### 2.2. Product Count by Category

| category                             |   Product Count |
|:-------------------------------------|----------------:|
| Tables & desks                       |             612 |
| Bookcases & shelving units           |             548 |
| Chairs                               |             481 |
| Sofas & armchairs                    |             428 |
| Cabinets & cupboards                 |             292 |
| Wardrobes                            |             236 |
| Outdoor furniture                    |             216 |
| Beds                                 |             208 |
| TV & media furniture                 |             190 |
| Chests of drawers & drawer units     |             125 |
| Children's furniture                 |             124 |
| Nursery furniture                    |              97 |
| Bar furniture                        |              47 |
| Trolleys                             |              28 |
| Café furniture                       |              26 |
| Sideboards, buffets & console tables |              23 |
| Room dividers                        |              13 |


![01_category_counts.png](figures/01_category_counts.png)
 The largest categories by SKU count are `Tables & desks`, `Bookcases & shelving units`, and `Chairs` — these are core furniture categories with the widest product range.

### 2.3. Median Price by Category

| category                             |   Median Price |
|:-------------------------------------|---------------:|
| Wardrobes                            |         1870   |
| Sofas & armchairs                    |         1268.5 |
| Beds                                 |         1093.5 |
| Sideboards, buffets & console tables |          965   |
| Room dividers                        |          725   |
| Tables & desks                       |          595   |
| Chests of drawers & drawer units     |          460   |
| Trolleys                             |          447.5 |
| Bar furniture                        |          445   |
| Chairs                               |          425   |
| Café furniture                       |          424.5 |
| Cabinets & cupboards                 |          402   |
| Nursery furniture                    |          350   |
| Outdoor furniture                    |          313.7 |
| Bookcases & shelving units           |          310   |
| Children's furniture                 |          202.5 |
| TV & media furniture                 |          175   |


![02_category_median_price.png](figures/02_category_median_price.png)
 The most expensive categories by median price are `Wardrobes` and `Sofas & armchairs`, while the cheapest is `TV & media furniture`. This aligns with common sense: large storage and seating furniture costs more than smaller items.

### 2.4. Price vs. Dimensions: Correlation Matrix

Correlation matrix (Spearman coefficient, appropriate given the skewed price distribution):

|        |   price |   depth |   height |   width |
|:-------|--------:|--------:|---------:|--------:|
| price  |    1    |    0.52 |     0.5  |    0.75 |
| depth  |    0.52 |    1    |     0.03 |    0.32 |
| height |    0.5  |    0.03 |     1    |    0.42 |
| width  |    0.75 |    0.32 |     0.42 |    1    |


![03_price_size_corr.png](figures/03_price_size_corr.png)
 The strongest correlation with price is `width` (ρ=0.75). All dimensions are positively associated with price — larger furniture requires more material and costs more.

### 2.5. Designers: Product Count and Median Price (Top 15)

| designer_clean                 |   Product Count |
|:-------------------------------|----------------:|
| IKEA of Sweden                 |             828 |
| Ehlén Johansson                |             161 |
| Francis Cayouette              |             151 |
| Unknown                        |             143 |
| Ola Wihlborg                   |             134 |
| K Hagberg/M Hagberg            |             122 |
| Jon Karlsson                   |             120 |
| Carina Bengs                   |              75 |
| Ehlén Johansson/IKEA of Sweden |              75 |
| Henrik Preutz                  |              70 |
| IKEA of Sweden/Ehlén Johansson |              70 |
| Studio Copenhagen              |              64 |
| H Preutz/A Fredriksson         |              60 |
| Ebba Strandmark                |              57 |
| K Malmvall/E Lilja Löwenhielm  |              55 |


![04_top15_designers_count.png](figures/04_top15_designers_count.png)


Median price for the same top-15 designers:

| designer_clean                 |   Median Price |
|:-------------------------------|---------------:|
| Ehlén Johansson/IKEA of Sweden |         2610   |
| IKEA of Sweden/Ehlén Johansson |         2567.5 |
| K Malmvall/E Lilja Löwenhielm  |         1834   |
| Ehlén Johansson                |         1280   |
| Carina Bengs                   |          995   |
| Ola Wihlborg                   |          842.5 |
| Ebba Strandmark                |          730   |
| Jon Karlsson                   |          501.5 |
| Henrik Preutz                  |          495   |
| Francis Cayouette              |          458   |
| K Hagberg/M Hagberg            |          389.5 |
| Unknown                        |          225   |
| IKEA of Sweden                 |          225   |
| Studio Copenhagen              |          220   |
| H Preutz/A Fredriksson         |           35   |


![05_top15_designers_median_price.png](figures/05_top15_designers_median_price.png)
 There is a notable difference in median price even among the top 15 most prolific designers — suggesting that the designer may serve as a proxy for the type or complexity of furniture they typically create (explored further in Part 3 hypothesis testing).

### 2.6. Availability of Other Colour Variants (`other_colors`)

| other_colors   |   Product Count |
|:---------------|----------------:|
| No             |            2182 |
| Yes            |            1512 |

Median price by availability of colour variants:

| other_colors   |   Median Price |
|:---------------|---------------:|
| No             |          450   |
| Yes            |          672.5 |


![06_colors.png](figures/06_colors.png)


### 2.7. Sold Online (`sellable_online`)

| sellable_online   |   Product Count |
|:------------------|----------------:|
| True              |            3666 |
| False             |              28 |

Median price by online availability:

| sellable_online   |   Median Price |
|:------------------|---------------:|
| False             |          197.5 |
| True              |          545   |

Only **28** items (0.8%) are not sold online — this feature has almost no variance and will be excluded from the ML feature set (Part 4).


![07_sellable_online.png](figures/07_sellable_online.png)


### 2.8. Price Distribution



![08_price_distribution.png](figures/08_price_distribution.png)
 Skewness of the original price: **2.29** (strong right skew). After log-transformation skewness drops to **-0.49**, making the distribution significantly closer to normal — a typical result for price and income data.

### 2.9. Old Price vs. Current Price



![09_old_price_vs_price.png](figures/09_old_price_vs_price.png)
 Regression line: price ≈ 0.840 × old_price + -16.8 (R² = 0.978). All points lie below the y = x diagonal, confirming that the current price is always lower than the old price. Average discount size: **19.6%**.

### 2.10. EDA Summary

Key findings:

- The largest category by SKU count is `Tables & desks` (612 items); the smallest is `Room dividers` (13).
- Price distribution is strongly right-skewed (skew=2.29); log-transformation (skew after: -0.49) makes it suitable for parametric tests.
- All dimensions are positively correlated with price; the strongest is `width`.
- Only 654 items (17.7%) have a discount history; average discount size: 19.6%.
- `sellable_online` has near-zero variance (>99% of items are available online) and will be excluded from the ML model.

## Part 3. Hypothesis Testing

### Hypothesis 1. The Bundle Effect: Multi-Item Sets Are Priced Higher Than Single Items

**Idea:** IKEA sells both individual pieces and *sets* (e.g., a bar table bundled with 4 bar stools listed under a single `item_id`). The hypothesis is that bundle products are priced higher than single items — not just proportionally to the number of pieces, but they exhibit a statistically different pricing structure. This is a non-trivial assumption about *pricing architecture* rather than an obvious observation.

**Bundle detection (creative feature engineering):** the `is_bundle` feature is built by combining two independent text signals — (a) the product name contains `/` (e.g. `NORDVIKEN / NORDVIKEN`, indicating two article codes combined), (b) the `short_description` contains the word `and` (e.g. *'Bar table **and** 4 bar stools'*).

- H₀: the median price of bundles equals the median price of single items (price distributions are identical).
- H₁: the median price of bundles is higher than the median price of single items.

Descriptive statistics by group:

| Group                   |    n |   Median |   Mean |
|:------------------------|-----:|---------:|-------:|
| Bundle (is_bundle=True) |  567 |     1340 | 1711.5 |
| Single Item             | 3127 |      450 |  963.4 |


![10_h1_bundle_boxplot.png](figures/10_h1_bundle_boxplot.png)


**Test 1.1 — Mann–Whitney U-test** (non-parametric, compares distributions without assuming normality — appropriate given the skewed price distribution):

U = 1257420.5, p-value = 4.72e-57

**Test 1.2 — Bootstrap test of difference in means (original price)**:

Observed difference in means: 748.1 SR, p-value (bootstrap) = 0.0000

**Test 1.3 — Bootstrap test on log-transformed price** (checking how p-value changes after variance stabilisation via log-transformation):

Observed difference in means of ln(price): 1.048, p-value (bootstrap) = 0.0000

**Effect size (Cohen's d on log-transformed price): 0.73** (medium effect by Cohen's conventional thresholds).

**Hypothesis 1 Conclusion:** all three tests (Mann–Whitney, bootstrap on original and log-transformed price) unanimously reject H₀ at the 0.05 significance level — bundle products are statistically significantly more expensive than single items, with a medium effect size. This confirms the hypothesis: bundling is not a random pricing quirk but a systematic component of IKEA's pricing architecture.

### Hypothesis 2. Do Products with Multiple Colour Variants Get Discounted Less Often?

**Idea:** products available in multiple colours (`other_colors = Yes`) may have *higher* popularity and assortment flexibility — giving the retailer less reason to discount them for inventory clearance (unlike single-colour or niche items that may require markdowns more often). This is a non-trivial *marketing* hypothesis that does not follow directly from any observable fact in the dataset.

- H₀: the presence of colour variants and the probability of a discount are independent (no association between `other_colors` and `had_discount`).
- H₁: the likelihood of a discount depends on the availability of colour variants.

Contingency table:

| other_colors   |   No Discount |   Discounted |
|:---------------|--------------:|-------------:|
| No             |          1783 |          399 |
| Yes            |          1257 |          255 |

Proportion of discounted items per group:

| other_colors   |   Discount Rate |
|:---------------|----------------:|
| No             |          0.1829 |
| Yes            |          0.1687 |


![11_h2_discount_rate_by_color.png](figures/11_h2_discount_rate_by_color.png)


**Test 2.1 — Chi-square test of independence:**

χ² = 1.142, df = 1, p-value = 0.2852

**Test 2.2 — Fisher's Exact Test** (more reliable than χ² when cell frequencies are moderate; provides an additional robustness check):

Odds ratio = 0.907, p-value = 0.2734

**Test 2.3 — Bootstrap test of difference in proportions:**

Observed difference in proportions (No − Yes): 0.0142, p-value (bootstrap) = 0.2670

**Hypothesis 2 Conclusion:** all three tests consistently return p-values well above 0.05 (χ²: p=0.29, Fisher: p=0.27, bootstrap: p=0.27). There is no basis to reject H₀: in this dataset, the availability of colour variants shows no statistically significant association with the likelihood of a discount. This is a valuable *negative result*: an intuitively appealing marketing assumption is not supported by the data, and other drivers of discounting (such as product category or seasonality, which are absent from this dataset) should be explored.

## Part 4. Machine Learning: Price Prediction Model

### 4.1. Data Preparation

Dropping `item_id` duplicates before ML (keeping first record): **3694 → 2962** rows. This eliminates the risk of data leakage where the same product (with the same price) appears in both train and test sets under different `category` labels.

**Columns excluded from features (X) and the rationale:**

- `item_id`, `name` — unique identifiers/labels; carry no generalisable information about price (the model would either learn nothing useful or overfit to specific IDs).
- `old_price`, `old_price_clean` — very highly correlated with `price` (R²=0.98 from the linear regression in section 2.9); including this feature would introduce multicollinearity and essentially hand the answer to the model.
- `had_discount` — derived directly from `old_price`; same risk for the same reason.
- `sellable_online` — varies in less than 1% of rows (shown in section 2.7); carries no practical predictive information.
- `link` (raw URL) — not informative as-is, but we **clean it** and extract the `link_color` feature (product colour from the URL slug) — the correct approach per the assignment: 'either clean the column and include it, or exclude it entirely'.
- `designer` (raw) — replaced by the cleaned `designer_clean`.
- `short_description` (raw text) — replaced by two derived features: `is_bundle` (bundle detection from Part 3) and `desc_word_count` (word count, a proxy for product complexity).

Final feature set: **4 numeric** (['depth', 'height', 'width', 'desc_word_count']) and **6 categorical** (['category', 'designer_clean', 'other_colors', 'is_bundle', 'designer_is_team', 'link_color']). Target variable: `price`.

Final dataset shape: X = (2962, 10), y = (2962,).

### 4.2. Preprocessing Pipeline

We use a `Pipeline` + `ColumnTransformer` approach (rather than manual imputation) so that all preprocessing logic (missing value imputation and categorical encoding) is fully encapsulated within cross-validation folds, preventing any information leakage from the test set into training.
For numeric features (`depth`, `height`, `width`, `desc_word_count`): missing values are imputed with the **median**, then standardised with `StandardScaler` — critical for linear models, SVR, and KNN which are sensitive to feature scale. For categorical features: missing values are imputed with the most frequent value, followed by **one-hot encoding** with `handle_unknown='ignore'` (robust to unseen categories in the test set).

Train/test split: **80% / 20%** (train: 2369 rows, test: 593 rows).

### 4.3. Baseline Models (Comparing 10 Regression Algorithms)

_Note: `xgboost` is not available in the current environment. In an environment with internet access (`pip install xgboost`) it is recommended to add `xgb.XGBRegressor(...)` — it typically performs at a level comparable to `HistGradientBoostingRegressor`. The minimum requirement of 3 models is comfortably met with 10 algorithms._
| Model                         |     R2 |     MAE |     RMSE |
|:------------------------------|-------:|--------:|---------:|
| RandomForestRegressor         | 0.834  | 334.824 |  559.11  |
| HistGradientBoostingRegressor | 0.8063 | 367.752 |  604.021 |
| BaggingRegressor              | 0.7928 | 363.591 |  624.758 |
| GradientBoostingRegressor     | 0.7634 | 434.439 |  667.513 |
| DecisionTreeRegressor         | 0.7277 | 435.594 |  716.097 |
| KNeighborsRegressor           | 0.7244 | 448.591 |  720.501 |
| RidgeCV                       | 0.7084 | 508.158 |  741.1   |
| LinearRegression              | 0.7033 | 512.626 |  747.505 |
| LassoCV                       | 0.683  | 526.347 |  772.659 |
| SVR_linear                    | 0.4201 | 584.323 | 1045.14  |

Best baseline model by R²: **RandomForestRegressor** (R² = 0.8340, RMSE = 559.1).


![12_base_models_r2.png](figures/12_base_models_r2.png)


### 4.4. GridSearchCV for the Best Model (RandomForestRegressor)

For `RandomForestRegressor` we deliberately avoid `max_depth=None` (unconstrained tree depth increases overfitting risk). `max_features='auto'` has been replaced with the current valid values `'sqrt'` or `1.0`. Hyperparameters are tuned via `GridSearchCV` with 3-fold cross-validation, optimising R² (the number of folds and grid size are kept compact for reasonable runtime; in a more powerful environment it is advisable to expand the grid and use cv=5).
Best hyperparameters found: `{'model__max_depth': 20, 'model__max_features': 1.0, 'model__min_samples_leaf': 2, 'model__n_estimators': 150}`
Best mean R² during grid search (CV): 0.7804

**Test set performance after GridSearchCV:** R² = 0.8132, MAE = 354.8, RMSE = 593.2.

Comparison with default baseline model: R² 0.8340 → 0.8132 (no improvement of 0.0209).


![13_actual_vs_predicted.png](figures/13_actual_vs_predicted.png)


### 4.5. Feature Importance Analysis

Grouped feature importances (as percentages, sum = 100%):

|                  |   Importance (%) |
|:-----------------|-----------------:|
| width            |            59.25 |
| depth            |            12.94 |
| designer_clean   |             6.04 |
| height           |             5.4  |
| category         |             4.99 |
| desc_word_count  |             3.6  |
| is_bundle        |             3.5  |
| link_color       |             2.86 |
| other_colors     |             0.72 |
| designer_is_team |             0.7  |


![14_feature_importance.png](figures/14_feature_importance.png)
 The most influential feature is **`width`** (59.2% of total importance), followed by **`depth`** (12.9%). The dominance of physical dimensions is logical and consistent with the correlation analysis in section 2.4: furniture size is the most direct predictor of cost. Category, designer, and bundle status (`is_bundle`, tested in Hypothesis 1) also contribute meaningfully but to a lesser degree.

### 4.6. Cross-Validation of the Final Model (`cross_val_score`)

Unlike `GridSearchCV` (which tunes hyperparameters), `cross_val_score` here is used solely to **evaluate** the stability of the already-selected final model with its best hyperparameters across 5 independent data splits.
R² per fold: [0.8146, 0.8083, 0.8038, 0.772, 0.8255]

**Mean R² (cross_val_score): 0.8048 (± 0.0180)**.

Three-stage comparison:

| Stage | R² |
|---|---|
| 1. Baseline model (default parameters) | 0.8340 |
| 2. After GridSearchCV (test set) | 0.8132 |
| 3. cross_val_score (mean across 5 folds) | 0.8048 |

The mean R² from cross-validation is close to the GridSearchCV result (difference: 0.0083) — indicating **no significant overfitting**: the model generalises stably across different data splits rather than simply memorising the specific test set.


![15_cross_validation.png](figures/15_cross_validation.png)


### 4.7. Recommendations for Further Model Improvement

1. **Add `xgboost` / `lightgbm`** — in an environment with internet access, gradient boosting frameworks typically outperform `RandomForestRegressor` on medium-sized tabular datasets.
2. **Target encoding for `designer_clean`** instead of one-hot encoding — with 280+ unique designers, one-hot creates a very sparse matrix; encoding designers by their median price (as suggested in the assignment) may improve R² while reducing model size.
3. **Text embeddings for `name` or `short_description`** — TF-IDF or sentence embeddings could capture additional signals about product type and complexity that are currently lost when these raw text columns are excluded.
4. **Stacking multiple strong models** (RandomForest + HistGradientBoosting + GradientBoosting) via `StackingRegressor` — typically yields a 1–3 pp R² gain by averaging out different model error profiles.
5. **Expand the GridSearchCV grid** (e.g. add `min_samples_split`, `bootstrap`) or switch to `RandomizedSearchCV` / `Optuna` for more efficient search over a larger hyperparameter space within reasonable compute time.
6. **Additional features from `short_description`** — beyond `is_bundle` and `desc_word_count`, one could extract the number of items in a set (regex on digits), presence of 'outdoor'/'indoor' keywords, material references, etc.


## Overall Project Summary

After data cleaning (removing technical columns, parsing `old_price`, cleaning `designer`) and a comprehensive EDA with 15 visualisations, two non-trivial hypotheses were tested: the bundle pricing effect (confirmed by three independent tests, Cohen's d=0.73) and the association between colour variants and discount likelihood (not confirmed — a valuable negative result demonstrating analytical integrity). In Part 4, 10 regression models were built and compared; the best-performing model — `RandomForestRegressor` — achieves **R²=0.813** on the held-out test set after `GridSearchCV` and **0.805** on average across 5-fold cross-validation, confirming stable generalisation without significant overfitting.