# FEATURE ENGINEERING
# DS -> Data Prep -> Feature Engineering

## Where It Fits
```
Data Science
│
├── Data Preprocessing & Feature Engineering
│   ├── Data Cleaning
│   │   ├── Missing Values                         (Stats)
│   │   ├── Outliers                               (Stats)
│   │   └── Duplicate / Inconsistent Data          (Stats)
│   │
│   ├── Feature Engineering                        <--- YOU ARE HERE
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

> Feature Engineering transforms clean data into the best possible input for a model. The model can only work with what you give it — better features almost always beat a better algorithm.

---

## The 3 Core Concepts

### 1. What a Feature Is
> The raw material every ML model works with.

- **Feature (X)** — Any input column used to make a prediction (Size, Age, Income).
- **Target (Y)** — The output column the model predicts (Price, Defaulted, Passed).
- **Raw Feature** — A column exactly as it arrived in the dataset.
- **Engineered Feature** — A column you created or transformed from raw data to expose a pattern more clearly.

```
Raw:       Size = 1500,  Year_Built = 1990
Engineered: Price_per_sqft = Price / Size
            House_Age = 2024 − Year_Built = 34
```

---

### 2. Why Models Can't Do This Themselves
> ML models are good at finding patterns in numbers — but only in the features you provide.

- A Linear Regression can't multiply two columns — you have to create the interaction term first.
- A model sees `Year_Built = 1990` as a large number, not as "34 years old."
- A model sees `Neighborhood = "Urban"` as text — unusable until encoded as numbers.

Feature Engineering is the step where domain knowledge gets baked into the data.

---

### 3. Signal vs. Noise
> The core tradeoff when adding new features.

- **Signal** — A feature that genuinely correlates with the target and helps the model predict better.
- **Noise** — A feature that adds random variation and causes the model to overfit.
- **Curse of Dimensionality** — As the number of features grows, the data becomes increasingly sparse and models get harder to train reliably.

> More features is not always better. Every feature you add should earn its place by improving validation performance.

---

## Engineering Types

### Scaling & Normalization
> Adjusting the range of numeric features so no single feature dominates due to its units.

**When it matters:** Any algorithm that uses distance or gradient descent — k-NN, SVM, Linear Regression, Neural Networks. Tree-based models (Decision Trees, Random Forest, XGBoost) do **not** need scaling — they split on thresholds, not distances.

**Two main methods:**

**Min-Max Scaling (Normalization):** Squashes all values into [0, 1].
```
X_scaled = (X − X_min) / (X_max − X_min)
```

**Standardization (Z-score scaling):** Centers around mean = 0, std = 1.
```
X_scaled = (X − mean) / std_deviation
```

**Example:** Why unscaled features break distance-based models.

| House | Size (sqft) | Bedrooms | Price    |
|-------|-------------|----------|----------|
| A     | 1,500       | 3        | $300,000 |
| B     | 2,000       | 4        | $450,000 |
| C     | 1,200       | 2        | $200,000 |

Size ranges ~1,000–2,500. Bedrooms ranges 2–4. In k-NN, the distance between House A and B is dominated entirely by Size differences (~500) vs. Bedroom differences (~1). Bedrooms becomes invisible.

After standardization:

| House | Size (scaled) | Bedrooms (scaled) |
|-------|---------------|-------------------|
| A     | 0.27          | 0.00              |
| B     | 1.37          | 1.22              |
| C     | −0.82         | −1.22             |

Both features now contribute on equal footing.

| | Min-Max | Standardization |
|---|---|---|
| **Output range** | [0, 1] | No fixed range (mean=0, std=1) |
| **Affected by outliers** | Yes — one extreme distorts the scale | Less so |
| **Best for** | Neural networks, pixel values | Most other algorithms |
| **Loses meaning of zero** | Yes | No |

> **Rule of thumb:** Default to Standardization. Use Min-Max when the algorithm explicitly requires inputs in [0, 1].

---

### Encoding Categorical Variables
> Converting text categories into numbers so models can use them.

**When it matters:** Always — no ML algorithm accepts raw text. The method matters because wrong encoding introduces false relationships.

**Three main methods:**

**Label Encoding:** Assigns an integer to each category.
```
Rural → 0,  Suburban → 1,  Urban → 2
```

Implies an ordering: Urban (2) > Suburban (1) > Rural (0). This is wrong unless the categories are genuinely ordinal. Only use for truly ordered categories.

```
✅ Ordinal (use label encoding):  Small=0, Medium=1, Large=2
❌ Nominal (don't):               Rural=0, Suburban=1, Urban=2
```

**One-Hot Encoding:** Creates one binary column per category.

| House | is_Rural | is_Suburban | is_Urban |
|-------|----------|-------------|----------|
| A     | 0        | 1           | 0        |
| B     | 0        | 0           | 1        |
| C     | 1        | 0           | 0        |
| D     | 0        | 0           | 1        |

No false ordering. Each category is independent. Best for most ML models.

> **Dummy variable trap:** In regression, drop one column (e.g. is_Rural) to avoid perfect multicollinearity — when all three columns are present, one is always predictable from the other two. Keep k−1 columns for k categories.

**Target Encoding:** Replaces each category with the mean target value for that category.
```
Urban houses average $425,000     → Urban    = 425,000
Suburban houses average $310,000  → Suburban = 310,000
Rural houses average $200,000     → Rural    = 200,000
```

Powerful — captures the relationship between category and target directly. But risks data leakage if computed on the full dataset.

```
WRONG: compute means on full dataset → encode → split
RIGHT: split first → compute means on train only → encode train and test
```

| | Label Encoding | One-Hot Encoding | Target Encoding |
|---|---|---|---|
| **Implies order** | Yes | No | No |
| **Dimensionality** | +0 columns | +k columns | +0 columns |
| **Leakage risk** | No | No | Yes (if not careful) |
| **Best for** | Ordinal categories | Nominal, low cardinality | High cardinality (many categories) |

---

### Creating New Features
> Building new columns from existing ones to expose patterns the model can't find on its own.

**When it matters:** When the raw features don't directly capture what drives the target. Models find linear relationships easily — but many real patterns are ratios, interactions, or derived values.

**Types of new features:**

**Interaction Terms:** Multiply two features together to capture joint effects.
```
Size × Bedrooms  →  captures "large house with many rooms"
Income × Age     →  captures wealth trajectory
```

**Ratio Features:** Divide one feature by another to create a density or rate.
```
Price / Size          → Price per sqft
Bedrooms / Bathrooms  → Layout ratio
Income / Debt         → Debt-to-income ratio
```

**Derived Features:** Extract new meaning from existing columns.
```
House_Age        = Current_Year − Year_Built
Days_Since_Last  = Today − Last_Purchase_Date
Is_Weekend       = 1 if Day ∈ {Sat, Sun} else 0
```

**Binning:** Convert continuous values into discrete buckets.
```
Age:
  0–17   → "Minor"
  18–35  → "Young Adult"
  36–60  → "Adult"
  60+    → "Senior"
```

Useful when the relationship between a feature and the target is non-linear and step-like rather than smooth.

**Polynomial Features:** Raise existing features to higher powers (links to Polynomial Regression).
```
Size²   → captures diminishing returns on price per sqft
Age³    → captures accelerating decay in property value
```

**Example:** Full feature engineering on the housing dataset.

| House | Size | Year_Built | Bedrooms | Bathrooms | Price    |
|-------|------|------------|----------|-----------|----------|
| A     | 1500 | 1990       | 3        | 2         | $300,000 |
| B     | 2000 | 2005       | 4        | 3         | $450,000 |
| C     | 1200 | 1975       | 2        | 1         | $200,000 |

After engineering:

| House | Size | House_Age | Bed_Bath_Ratio | Price_per_sqft | Size²     |
|-------|------|-----------|----------------|----------------|-----------|
| A     | 1500 | 34        | 1.5            | $200           | 2,250,000 |
| B     | 2000 | 19        | 1.33           | $225           | 4,000,000 |
| C     | 1200 | 49        | 2.0            | $167           | 1,440,000 |

The model now has direct access to `House_Age` instead of having to figure out that 1990 means "older" — and `Price_per_sqft` as a pre-computed signal it would have taken many steps to approximate.

> **Watch out for:** Creating too many features causes overfitting and slows training. Always validate that a new feature improves performance on the validation set before keeping it.

---

## The Feature Engineering Checklist

```
Clean Data Received
│
├── 1. Encode categorical variables
│       - Ordinal → Label Encoding
│       - Nominal → One-Hot Encoding
│       - High cardinality → Target Encoding (train only)
│
├── 2. Scale numeric features
│       - Default: Standardization
│       - Neural nets / [0,1] required: Min-Max
│       - Tree-based models: skip
│
├── 3. Create new features
│       - Interaction terms
│       - Ratios and rates
│       - Derived / extracted values
│       - Binning
│       - Polynomial terms
│
├── 4. Validate each new feature
│       - Does it improve validation performance?
│       - Is it correlated with existing features (redundant)?
│
└── Engineered data → ready for Feature Selection
```

---

> **Key takeaway:** Feature Engineering is where domain knowledge meets math. A model can only find patterns in the numbers you give it — the better you represent the problem, the less work the model has to do.
