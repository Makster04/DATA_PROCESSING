# DATA PREPROCESSING & FEATURE ENGINEERING
# DS -> Data Prep -> Clean, Transform, Select

## Where It Fits
```
Data Science
│
├── Data Preprocessing & Feature Engineering   <--- YOU ARE HERE
│   ├── Data Cleaning
│   │   ├── Missing Values
│   │   ├── Outliers
│   │   └── Duplicate / Inconsistent Data
│   │
│   ├── Feature Engineering
│   │   ├── Scaling & Normalization
│   │   ├── Encoding Categorical Variables
│   │   └── Creating New Features
│   │
│   ├── Feature Selection
│   │   ├── Filter Methods
│   │   ├── Wrapper Methods
│   │   └── Embedded Methods
│   │
│   └── Data Splitting
│       ├── Train / Validation / Test Split
│       └── Cross-Validation
│
└── Machine Learning
    ├── Supervised Learning
    ├── Unsupervised Learning
    ├── Reinforcement Learning
    └── Ensemble Methods
```

---

> This step comes **before** any Machine Learning. Every algorithm in the ML docs assumes clean, well-structured data — but raw data is almost never that. Garbage in, garbage out.

---

## The 3 Core Concepts

### 1. Why Raw Data Is Never Ready
> The state of data when you first receive it vs. what ML models actually need.

- **Structured Data** — Rows and columns (CSV, database tables). Still likely has missing values, wrong types, and outliers.
- **Unstructured Data** — Text, images, audio. Needs to be converted into numbers entirely before any ML algorithm can touch it.
- **The Golden Rule** — Models are mathematical functions. Everything fed into them must be a number, complete, and on a sensible scale.

```
Raw data problems:
  ├── Missing values      → model crashes or learns wrong patterns
  ├── Wrong scale         → one feature dominates all others
  ├── Categorical text    → model can't compute with "Suburban"
  ├── Outliers            → one extreme value skews the whole model
  └── Irrelevant features → noise that hurts more than it helps
```

---

### 2. The Pipeline Concept
> Preprocessing steps must be applied in a consistent order — and the same transformations used on training data must be applied identically to new data.

- **Pipeline** — A fixed sequence of preprocessing steps chained together so they can be applied consistently to any dataset.
- **Data Leakage** — When information from the test set accidentally influences preprocessing (e.g. computing the mean of the whole dataset before splitting). A silent killer of model validity.

```
WRONG (data leakage):
  Scale entire dataset → split into train/test

CORRECT:
  Split into train/test → fit scaler on train only → apply to both
```

---

### 3. Feature vs. Target
> A reminder of the vocabulary used throughout preprocessing.

- **Feature (X)** — Any input column used to make predictions (Size, Age, Income).
- **Target (Y)** — The output column the model predicts (Price, Defaulted, Passed).
- **Feature Engineering** — Creating or transforming features to make patterns easier for the model to find.
- **Dimensionality** — The number of features. Too many can hurt — more features means more noise, longer training, and harder-to-interpret models.

---

## Data Cleaning

### Missing Values
> Deciding what to do when a cell in your dataset is blank.

**When it matters:** Almost always. Real-world data has gaps — sensors fail, forms go unfilled, records get corrupted.

**The three strategies:**

| Strategy | What it does | When to use |
|---|---|---|
| **Drop** | Remove rows or columns with missing data | When < 5% of data is missing and it's random |
| **Impute (Mean/Median/Mode)** | Fill with a summary statistic | Numeric columns with moderate missingness |
| **Impute (Model-based)** | Predict the missing value using other features | When missingness is high or not random |

**Example:** A housing dataset with missing values.

| House | Size | Bedrooms | Price    |
|-------|------|----------|----------|
| A     | 1500 | 3        | $300,000 |
| B     | 2000 | NaN      | $450,000 |
| C     | NaN  | 2        | $200,000 |
| D     | 1800 | 3        | $400,000 |

```
# Option 1 — Drop rows with any missing value:
  Keeps: A, D   Loses: B, C  ← loses 50% of data, bad

# Option 2 — Mean imputation:
  Bedrooms mean = (3+2+3)/3 = 2.67 → round to 3
  Size mean = (1500+2000+1800)/3 = 1767

  Result:
  House B → Bedrooms = 3
  House C → Size = 1767
```

> **Watch out for:** Mean imputation hides the fact that data was missing. Adding a binary flag column `Bedrooms_was_missing` (1 = yes, 0 = no) preserves that signal for the model.

---

### Outliers
> Data points that sit far from the rest — either genuine extremes or data entry errors.

**When it matters:** Outliers disproportionately influence models that minimize squared error (Linear Regression, RMSE). Tree-based models are more robust.

**Detecting outliers — two methods:**

**IQR Method (Interquartile Range):**
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1

Outlier if: value < Q1 - 1.5×IQR  or  value > Q3 + 1.5×IQR
```

**Z-Score Method:**
```
Z = (value - mean) / std_deviation

Outlier if: |Z| > 3  (more than 3 standard deviations from mean)
```

**Example:** House prices dataset.

| House | Price      | Z-Score |
|-------|------------|---------|
| A     | $300,000   | 0.1     |
| B     | $450,000   | 0.8     |
| C     | $200,000   | −0.6    |
| D     | $380,000   | 0.5     |
| E     | $2,400,000 | 4.9 ← outlier |

House E's Z-score of 4.9 flags it as an outlier. Options: remove it, cap it at a ceiling value (winsorizing), or investigate if it's a data error.

> **Rule of thumb:** Don't automatically delete outliers. First ask — is this a data error, or a genuine extreme value? A $2.4M mansion is real; a 0-sqft house is probably an error.

---

### Duplicate & Inconsistent Data
> Rows that appear more than once, or values that represent the same thing but are formatted differently.

**When it matters:** Duplicates inflate your dataset and bias the model toward repeated patterns. Inconsistencies create phantom categories.

**Example:**

| Customer | City          |
|----------|---------------|
| A        | New York      |
| B        | new york      |
| C        | NY            |
| A        | New York      |  ← duplicate row

All four rows refer to the same city, but a model would treat them as four different values. Fixes:
- Standardize text: lowercase, strip whitespace, map aliases → `new york → New York`
- Drop exact duplicate rows
- Flag near-duplicates for manual review

---

## Feature Engineering

### Scaling & Normalization
> Adjusting the range of numeric features so no single feature dominates due to its units.

**When it matters:** Any algorithm that uses distance or gradient descent — k-NN, SVM, Linear Regression, Neural Networks. Tree-based models (Decision Trees, Random Forest, XGBoost) do **not** need scaling.

**Two main methods:**

**Min-Max Scaling (Normalization):** Squashes all values into [0, 1].
```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Standardization (Z-score scaling):** Centers around mean=0, std=1.
```
X_scaled = (X - mean) / std_deviation
```

**Example:** Why unscaled features cause problems.

| House | Size (sqft) | Bedrooms | Price    |
|-------|-------------|----------|----------|
| A     | 1,500       | 3        | $300,000 |
| B     | 2,000       | 4        | $450,000 |

Size ranges from 1,000–2,500. Bedrooms ranges from 1–5.
In a distance-based model, Size differences (~500) dwarf Bedroom differences (~1–2), making bedrooms nearly invisible. After standardization both features contribute equally.

| House | Size (scaled) | Bedrooms (scaled) |
|-------|---------------|-------------------|
| A     | −0.71         | −0.71             |
| B     | +0.71         | +0.71             |

> **Rule of thumb:** Use Standardization when your data has outliers (min-max gets distorted by them). Use Min-Max when you need values strictly between 0 and 1 (e.g. neural network inputs).

---

### Encoding Categorical Variables
> Converting text categories into numbers so models can use them.

**When it matters:** Always — no ML algorithm accepts raw text. The method you choose affects multicollinearity and model performance.

**Three main methods:**

**Label Encoding:** Assigns an integer to each category.
```
Rural → 0,  Suburban → 1,  Urban → 2
```
⚠️ Implies an order (Urban > Suburban > Rural) that may not exist. Only use when the category is genuinely ordinal (Small < Medium < Large).

**One-Hot Encoding:** Creates a binary column per category.

| House | is_Rural | is_Suburban | is_Urban |
|-------|----------|-------------|----------|
| A     | 0        | 1           | 0        |
| B     | 0        | 0           | 1        |
| C     | 1        | 0           | 0        |

No false ordering. Best for ML models. Drop one column to avoid multicollinearity in regression (dummy variable trap).

**Target Encoding:** Replaces each category with the mean target value for that category.
```
Urban houses average $425,000     → Urban → 425000
Suburban houses average $310,000  → Suburban → 310000
Rural houses average $200,000     → Rural → 200000
```
Powerful but risks data leakage — must be computed on training data only.

---

### Creating New Features
> Building new columns from existing ones to expose patterns the model can't find on its own.

**When it matters:** When the raw features don't directly capture what drives the target. A model can't multiply two columns together — you have to do it.

**Example:** Predicting house price.

```
# Raw features:
  Total_Rooms = 6,  Bathrooms = 2

# New features:
  Price_per_sqft  = Price / Size              ← density metric
  Bed_Bath_Ratio  = Bedrooms / Bathrooms      ← layout quality
  House_Age       = Current_Year - Year_Built ← derived from existing column
  Is_Large        = 1 if Size > 2000 else 0   ← binary flag
```

Each new feature gives the model a pre-computed signal that would have taken many splits or coefficients to approximate on its own.

> **Watch out for:** Creating too many features (feature explosion). More features = more noise = harder to train. Always validate that a new feature actually improves model performance.

---

## Feature Selection
> Choosing which features to keep and which to discard before training.

> The goal is to remove features that add noise without adding signal — reducing training time, improving accuracy, and making models more interpretable.

---

### Filter Methods
> Score each feature independently of the model, then keep the top-ranked ones.

**When to use:** Quick first pass to remove obviously irrelevant features before trying anything more expensive.

**Common metrics:**

| Method | Best for | What it measures |
|---|---|---|
| **Correlation** | Regression | Linear relationship between feature and target |
| **Chi-Squared** | Classification | Statistical dependence between categorical feature and target |
| **Variance Threshold** | Any | Drops features with near-zero variance (they carry no information) |

**Example:** Correlation filter for house price prediction.

| Feature      | Correlation with Price | Keep? |
|--------------|------------------------|-------|
| Size         | 0.85                   | ✅    |
| Bedrooms     | 0.72                   | ✅    |
| LotAge       | 0.08                   | ❌    |
| House_ID     | 0.01                   | ❌    |

LotAge and House_ID have near-zero correlation with price — drop them.

---

### Wrapper Methods
> Use the model itself to evaluate subsets of features — train, score, repeat with different combinations.

**When to use:** You want the best possible feature subset and can afford the computational cost.

**Common approaches:**

- **Forward Selection** — Start with no features. Add one at a time, keeping whichever improves the model most.
- **Backward Elimination** — Start with all features. Remove the least useful one at a time.
- **Recursive Feature Elimination (RFE)** — Train the model, rank features by importance, remove the weakest, retrain. Repeat until target number of features is reached.

**Example:** Backward Elimination on house price model.

```
Round 1: All 6 features → R² = 0.88
  Remove House_ID (least important) → R² = 0.88  ✅ no loss

Round 2: 5 features → R² = 0.88
  Remove LotAge → R² = 0.87  ✅ negligible loss

Round 3: 4 features → R² = 0.87
  Remove GarageSize → R² = 0.83  ❌ stop — meaningful drop
```

Final selected features: Size, Bedrooms, Bathrooms, Neighborhood.

---

### Embedded Methods
> Feature selection happens automatically during model training — the model learns which features matter.

**When to use:** You're already using a model that supports it (Lasso, Random Forest, XGBoost). The most efficient approach — selection and training happen in one step.

**Examples:**

- **Lasso Regression** — Drives irrelevant feature coefficients exactly to zero (covered in Supervised Learning doc).
- **Random Forest Feature Importance** — After training, ranks features by how much they reduced impurity across all trees.
- **XGBoost Feature Importance** — Ranks features by how often and how effectively they were used in splits.

**Example:** Random Forest feature importance scores.

| Feature      | Importance Score |
|--------------|-----------------|
| Size         | 0.42            |
| Neighborhood | 0.28            |
| Bedrooms     | 0.19            |
| Bathrooms    | 0.08            |
| LotAge       | 0.02            |
| House_ID     | 0.01            |

Drop LotAge and House_ID — they contribute almost nothing across the entire forest.

---

## Data Splitting

### Train / Validation / Test Split
> Dividing the dataset into separate portions for training, tuning, and final evaluation.

**Why it matters:** If you evaluate your model on the same data it trained on, you're testing whether it memorized the data — not whether it learned anything generalizable.

**The three sets:**

- **Training Set** — What the model learns from. Typically 60–70% of the data.
- **Validation Set** — Used to tune hyperparameters and compare models during development. Typically 10–20%.
- **Test Set** — Touched only once at the very end. The final honest measure of performance. Typically 10–20%.

```
Full Dataset (100%)
│
├── Training Set   (70%) ← model sees this
├── Validation Set (15%) ← you use this to tune
└── Test Set       (15%) ← locked away until the end
```

> **Critical rule:** Never tune your model based on test set performance. The moment you do, the test set becomes a second validation set and your final score is no longer trustworthy.

---

### Cross-Validation
> Instead of one fixed train/validation split, rotate through multiple splits and average the results.

**When to use:** Your dataset is small and a single split might be unrepresentative, or you want a more reliable estimate of model performance.

**k-Fold Cross-Validation:** Split data into k equal folds. Train on k-1 folds, validate on the remaining 1. Rotate k times so every fold gets a turn as the validation set.

**Example:** 5-Fold Cross-Validation on 100 houses.

```
Fold 1: Train on [2,3,4,5] → Validate on [1] → R² = 0.84
Fold 2: Train on [1,3,4,5] → Validate on [2] → R² = 0.87
Fold 3: Train on [1,2,4,5] → Validate on [3] → R² = 0.85
Fold 4: Train on [1,2,3,5] → Validate on [4] → R² = 0.83
Fold 5: Train on [1,2,3,4] → Validate on [5] → R² = 0.86

Average R² = 0.85  ← much more reliable than any single split
```

> **Rule of thumb:** Use 5-fold or 10-fold cross-validation as standard. The test set is still held out — cross-validation only replaces the validation set, never the test set.

---

## The Full Preprocessing Pipeline

### In the correct order:

```
Raw Data
│
├── 1. Remove duplicates & fix inconsistent values
│
├── 2. Handle missing values
│       (drop, mean impute, or model-based impute)
│
├── 3. Detect and handle outliers
│
├── 4. Split into Train / Validation / Test
│       ← MUST happen before any scaling or encoding
│
├── 5. Encode categorical variables
│       (fit on train, apply to all)
│
├── 6. Scale / normalize numeric features
│       (fit on train, apply to all)
│
├── 7. Create new features
│
├── 8. Select features
│       (filter, wrapper, or embedded)
│
└── Clean, transformed data ready for ML
```

> **Key takeaway:** Steps 5–8 must always be fit on the training set only, then applied to validation and test sets. Doing it any other way leaks information from the future into the past and makes your model look better than it really is.
