# DATA CLEANING
# DS -> Data Prep -> Data Cleaning

## Where It Fits
```
Data Science
│
├── Data Preprocessing & Feature Engineering
│   ├── Data Cleaning                              <--- YOU ARE HERE
│   │   ├── Missing Values                         (Stats)
│   │   ├── Outliers                               (Stats)
│   │   └── Duplicate / Inconsistent Data          (Stats)
│   │
│   ├── Feature Engineering
│   │   ├── Scaling & Normalization                (Linear Algebra + Stats)
│   │   ├── Encoding Categorical Variables         (Stats)
│   │   └── Creating New Features                  (Stats + Calculus)
│   │
│   ├── Feature Selection
│   │   ├── Filter Methods                         (Stats)
│   │   ├── Wrapper Methods                        (Stats + Calculus)
│   │   └── Embedded Methods                       (Calculus + Stats)
│   │
│   └── Data Splitting
│       ├── Train / Validation / Test Split        (Stats)
│       └── Cross-Validation                       (Stats)
│
└── Machine Learning
    ├── Supervised Learning
    │   ├── Regression
    │   │   ├── Linear Regression                  (Linear Algebra + Stats)
    │   │   ├── Polynomial Regression              (Calculus + Linear Algebra)
    │   │   └── Ridge / Lasso                      (Calculus + Linear Algebra)
    │   │
    │   └── Classification
    │       ├── Logistic Regression                (Calculus + Stats)
    │       ├── Decision Trees                     (Stats)
    │       ├── Random Forest                      (Stats)
    │       ├── SVM                                (Linear Algebra + Calculus)
    │       └── k-NN                               (Linear Algebra + Stats)
    │
    ├── Unsupervised Learning
    │   ├── Clustering
    │   │   ├── k-means                            (Linear Algebra + Stats)
    │   │   ├── Hierarchical Clustering            (Stats)
    │   │   └── DBSCAN                             (Linear Algebra + Stats)
    │   │
    │   ├── Dimensionality Reduction
    │   │   ├── PCA                                (Linear Algebra)
    │   │   └── t-SNE                              (Calculus + Stats)
    │   │
    │   └── Anomaly Detection
    │       └── Isolation Forest                   (Stats)
    │
    ├── Reinforcement Learning
    │   ├── Q-Learning                             (Calculus + Stats)
    │   └── Policy Gradient                        (Calculus + Stats)
    │
    └── Ensemble Methods
        ├── Bagging
        │   └── Random Forest                      (Stats)
        └── Boosting
            ├── XGBoost                            (Calculus + Stats)
            └── LightGBM                           (Calculus + Stats)
```

---

> Data Cleaning is the first step before any ML model is trained. Raw data is almost never ready — it has gaps, extremes, and inconsistencies that will break or mislead any algorithm downstream.

---

## The 3 Core Concepts

### 1. Why Raw Data Is Never Ready
> The state of data when you first receive it vs. what ML models actually need.

- **Structured Data** — Rows and columns (CSV, database tables). Still likely has missing values, wrong types, and outliers.
- **Unstructured Data** — Text, images, audio. Must be converted into numbers entirely before any ML algorithm can touch it.
- **The Golden Rule** — Models are mathematical functions. Everything fed into them must be a number, complete, and on a sensible scale.

```
Raw data problems:
  ├── Missing values        → model crashes or learns wrong patterns
  ├── Outliers              → one extreme value skews the whole model
  ├── Duplicate rows        → inflates dataset, biases toward repeated patterns
  └── Inconsistent labels   → "New York", "new york", "NY" treated as 3 categories
```

---

### 2. The Cost of Skipping It
> What actually happens downstream if data cleaning is skipped.

- **Missing values** → Most ML algorithms crash or silently substitute zeros, producing wrong predictions.
- **Outliers** → Linear Regression's least-squares objective gets pulled hard toward extreme values, distorting all coefficients.
- **Duplicates** → The model sees certain examples more than others and weights them too heavily.
- **Inconsistent categories** → One real category becomes many phantom categories, each with too little data to learn from.

---

### 3. Cleaning vs. Transforming
> An important distinction for knowing which step you're in.

- **Cleaning** — Fixing what's *wrong* with the data (missing, duplicate, corrupt). Done first.
- **Transforming** — Changing the *format* of valid data to suit the model (scaling, encoding). Done after cleaning.

---

## Cleaning Types

### Missing Values
> Deciding what to do when a cell in your dataset is blank.

**When it matters:** Almost always. Real-world data has gaps — sensors fail, forms go unfilled, records get corrupted.

**First, understand why data is missing:**

| Type | Meaning | Example |
|---|---|---|
| **MCAR** (Missing Completely At Random) | No pattern — missing by pure chance | Random sensor dropout |
| **MAR** (Missing At Random) | Missingness depends on other observed data | Older customers less likely to fill income field |
| **MNAR** (Missing Not At Random) | Missingness related to the missing value itself | High earners skipping the income question |

MNAR is the most dangerous — the missing values carry information you can't recover.

**The three strategies:**

| Strategy | What it does | When to use |
|---|---|---|
| **Drop** | Remove rows or columns with missing data | < 5% missing and pattern is random (MCAR) |
| **Mean / Median / Mode Imputation** | Fill with a summary statistic | Moderate missingness in numeric columns |
| **Model-Based Imputation** | Predict the missing value using other features | High missingness or MAR pattern |

**Example:** A housing dataset with missing values.

| House | Size | Bedrooms | Price    |
|-------|------|----------|----------|
| A     | 1500 | 3        | $300,000 |
| B     | 2000 | NaN      | $450,000 |
| C     | NaN  | 2        | $200,000 |
| D     | 1800 | 3        | $400,000 |

```
Option 1 — Drop rows with any missing value:
  Keeps: A, D   Loses: B, C  ← loses 50% of data, bad choice here

Option 2 — Median imputation:
  Bedrooms median = 3  → House B gets Bedrooms = 3
  Size median = 1,800  → House C gets Size = 1,800
```

**Adding a missingness flag:**

```
Before:                     After:
  Bedrooms = NaN     →        Bedrooms = 3,  Bedrooms_was_missing = 1
  Bedrooms = 3       →        Bedrooms = 3,  Bedrooms_was_missing = 0
```

The flag preserves the information that the value was imputed — the model can learn from the pattern of missingness itself.

> **Rule of thumb:** Never just impute and move on. Always add a `_was_missing` flag column alongside the imputed values.

---

### Outliers
> Data points that sit far from the rest — either genuine extremes or data entry errors.

**When it matters:** Models that minimize squared error (Linear Regression, RMSE) are disproportionately influenced by outliers. Tree-based models (Random Forest, XGBoost) are naturally more robust.

**Detecting outliers — two methods:**

**IQR Method (Interquartile Range):** Based on the middle spread of the data.
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 − Q1

Outlier if:  value < Q1 − 1.5 × IQR
         or  value > Q3 + 1.5 × IQR
```

**Z-Score Method:** Based on distance from the mean in standard deviations.
```
Z = (value − mean) / std_deviation

Outlier if: |Z| > 3
```

**Example:** House prices dataset.

| House | Price      | Z-Score     |
|-------|------------|-------------|
| A     | $300,000   | 0.1         |
| B     | $450,000   | 0.8         |
| C     | $200,000   | −0.6        |
| D     | $380,000   | 0.5         |
| E     | $2,400,000 | 4.9 ← outlier |

House E's Z-score of 4.9 flags it as an outlier. Three options:

| Option | What it does | When to use |
|---|---|---|
| **Remove** | Drop the row entirely | Confirmed data entry error |
| **Winsorize (Cap)** | Replace with the boundary value (e.g. 99th percentile) | Genuine extreme but don't want to lose the row |
| **Keep** | Leave it in | If the outlier is real and the model should know about it |

> **Rule of thumb:** Don't automatically delete outliers. First ask — is this a data error, or a genuine extreme value? A $2.4M mansion is real. A house with 0 sqft is probably an error.

---

### Duplicate & Inconsistent Data
> Rows that appear more than once, or values that represent the same thing formatted differently.

**When it matters:** Duplicates inflate your dataset and bias the model toward those repeated examples. Inconsistencies create phantom categories that fragment your data.

**Types of duplicates:**

| Type | Example | Fix |
|---|---|---|
| **Exact duplicate** | Two identical rows | Drop all but one |
| **Near-duplicate** | Same customer, slightly different entry | Flag for manual review or fuzzy match |
| **Logical duplicate** | Same entity, different ID | Merge on a shared key |

**Example — inconsistent categories:**

| Customer | City          |
|----------|---------------|
| A        | New York      |
| B        | new york      |
| C        | NY            |
| D        | New York      |
| A        | New York      | ← exact duplicate row

A model would see five rows but treat "New York", "new york", and "NY" as three separate cities — each with too little data to learn from.

**Cleaning steps:**
```
Step 1 — Standardize text:
  "new york" → "New York"
  "NY"       → "New York"

Step 2 — Drop exact duplicate rows:
  Customer A appears twice → keep one, drop the other

Step 3 — Result: 4 clean rows, all using "New York"
```

**Common inconsistency patterns to watch for:**

- Mixed case: `Male` vs `male` vs `MALE`
- Extra whitespace: `"Urban "` vs `"Urban"`
- Abbreviations: `"St."` vs `"Street"`, `"CA"` vs `"California"`
- Date formats: `2024-01-15` vs `15/01/2024` vs `Jan 15 2024`

---

## The Data Cleaning Checklist

```
Raw Data Received
│
├── 1. Audit the data
│       - How many rows and columns?
│       - What are the data types of each column?
│       - How many missing values per column?
│       - What does the distribution of each column look like?
│
├── 2. Handle missing values
│       - Identify MCAR / MAR / MNAR
│       - Drop, impute, or flag
│
├── 3. Detect and handle outliers
│       - IQR or Z-score method
│       - Remove, cap, or keep with justification
│
├── 4. Remove exact duplicates
│
├── 5. Standardize inconsistent values
│       - Text case, whitespace, abbreviations, date formats
│
└── Clean data → ready for Feature Engineering
```

---

> **Key takeaway:** Cleaning fixes what is *wrong*. Imputation doesn't create data — it makes assumptions. Every assumption you make should be documented, and every filled value should be flagged so the model can learn from the pattern of missingness, not just the filled number.
