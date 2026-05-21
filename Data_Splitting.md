# DATA SPLITTING
# DS -> Data Prep -> Data Splitting

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
│   └── Data Splitting                             <--- YOU ARE HERE
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

> Data Splitting is the last step before model training. It determines how you measure whether your model actually learned something — or just memorized your data.

---

## The 3 Core Concepts

### 1. The Fundamental Problem
> Why you can't evaluate a model on the same data it trained on.

- **Memorization vs. Learning** — A model can score perfectly on training data by memorizing every example without learning any generalizable pattern.
- **Overfitting** — When a model performs well on training data but poorly on new, unseen data.
- **Generalization** — The model's ability to perform well on data it has never seen before. This is the actual goal.

```
Evaluated on training data:
  Model sees 100 houses it already knows → R² = 0.98  ← meaningless

Evaluated on held-out test data:
  Model sees 20 houses it has never seen → R² = 0.84  ← the real score
```

---

### 2. Data Leakage
> When information from the future (test set) accidentally influences the model during training or preprocessing.

- **Leakage** — The model has access to information it shouldn't, making it look better than it really is.
- **Why it's silent** — The model trains and evaluates without error. You only discover leakage when the model fails in production.

```
Common leakage mistakes:
  ├── Scaling on full dataset before splitting
  ├── Imputing missing values using the full dataset mean
  ├── Target encoding using the full dataset
  └── Including future data in time-series models
```

The fix is always the same: **split first, then fit any preprocessing on training data only.**

---

### 3. The Three Sets and Their Roles
> Each portion of your data has a specific job, and mixing their roles destroys your evaluation.

| Set | Who uses it | Purpose |
|---|---|---|
| **Training** | The model | Learning patterns from labeled data |
| **Validation** | You | Tuning hyperparameters and comparing models |
| **Test** | Final evaluation only | One honest measure of real-world performance |

> The test set is touched exactly once — after all decisions have been made. If you look at it to make any decision, it becomes a second validation set and your reported performance is optimistic.

---

## Splitting Methods

### Train / Validation / Test Split
> Dividing the dataset into three fixed portions before any model training begins.

**When to use:** You have enough data that a fixed split gives representative samples in each portion. Standard for most ML projects.

**Typical proportions:**

```
Full Dataset (100%)
│
├── Training Set   (70%) ← model learns from this
├── Validation Set (15%) ← you tune on this
└── Test Set       (15%) ← touched once at the very end
```

Proportions vary by dataset size:

| Dataset Size | Recommended Split |
|---|---|
| Small (< 1,000 rows) | 60 / 20 / 20 — or use cross-validation instead |
| Medium (1k–100k rows) | 70 / 15 / 15 |
| Large (> 100k rows) | 80 / 10 / 10 — training data matters most |

**Example:** 1,000 house records split for a price prediction model.

```
Full dataset: 1,000 houses (shuffled randomly first)
│
├── Training Set:   700 houses  → model trains on these
├── Validation Set: 150 houses  → you test different hyperparameters here
└── Test Set:       150 houses  → final R² reported from this alone
```

**Why shuffle first?** If data was collected chronologically, the last 30% might all be from recent years — a very different distribution than the training set. Random shuffling ensures all three sets have similar distributions.

> **Exception — Time Series:** Never shuffle time series data. The past must train the model; the future must test it. Shuffling leaks future information into the past.

---

**The correct preprocessing order:**

```
WRONG (leakage):
  1. Compute mean on all 1,000 houses → impute missing values
  2. Fit scaler on all 1,000 houses → scale features
  3. Split into train / val / test

CORRECT:
  1. Split into train / val / test  ← split first, always
  2. Compute mean on 700 training houses only → impute all three sets
  3. Fit scaler on 700 training houses only → scale all three sets
```

---

### Cross-Validation
> Instead of one fixed train/validation split, rotate through multiple splits and average the results.

**When to use:** Your dataset is small and a single split might be unrepresentative, or you want a more reliable estimate of model performance during hyperparameter tuning.

**k-Fold Cross-Validation:** Split the training data into k equal folds. Train on k−1 folds, validate on the remaining 1. Rotate so every fold gets a turn as the validation set. Average the k scores.

**Example:** 5-Fold Cross-Validation on 500 training houses.

```
Fold 1: Train on folds [2,3,4,5] → Validate on fold [1] → R² = 0.84
Fold 2: Train on folds [1,3,4,5] → Validate on fold [2] → R² = 0.87
Fold 3: Train on folds [1,2,4,5] → Validate on fold [3] → R² = 0.85
Fold 4: Train on folds [1,2,3,5] → Validate on fold [4] → R² = 0.83
Fold 5: Train on folds [1,2,3,4] → Validate on fold [5] → R² = 0.86

Average R² = 0.85 ± 0.01  ← much more reliable than any single split
```

The ± 0.01 standard deviation tells you how stable the model is across different data subsets — a high variance here signals overfitting.

**Variants of Cross-Validation:**

| Variant | How it works | Best for |
|---|---|---|
| **k-Fold** | k equal folds, rotate | Standard choice — most datasets |
| **Stratified k-Fold** | Folds preserve class proportions | Imbalanced classification (e.g. 95% No, 5% Yes) |
| **Leave-One-Out (LOO)** | k = n, each sample is its own fold | Very small datasets (< 50 rows) |
| **Time Series Split** | Each fold only uses past data to predict future | Any time-ordered data |

**Time Series Split example — no shuffling:**

```
Fold 1: Train on months [Jan–Mar]  → Validate on [Apr]
Fold 2: Train on months [Jan–Apr]  → Validate on [May]
Fold 3: Train on months [Jan–May]  → Validate on [Jun]
```

The training window always ends before the validation window begins — mimicking how the model will actually be used in production.

---

### Cross-Validation vs. Single Split — Key Differences

| | Single Split | Cross-Validation |
|---|---|---|
| **Speed** | Fast (train once) | Slower (train k times) |
| **Reliability** | Depends on one split | Averaged over k splits |
| **Best for** | Large datasets | Small-to-medium datasets |
| **Variance estimate** | None | Yes (std dev across folds) |
| **Replaces test set?** | No | No — test set still held out |

> Cross-validation only replaces the validation set. The test set is always held out separately, regardless of whether you use a single split or cross-validation for tuning.

---

## Full Worked Example

**Setup:** 400 student records. Predicting whether a student passes an exam.

**Step 1 — Initial split (before any preprocessing):**

```
Full dataset: 400 students (shuffled)
│
├── Training Set:   280 students (70%)
├── Validation Set:  60 students (15%)
└── Test Set:        60 students (15%)  ← locked away
```

**Step 2 — Preprocess using training data only:**

```
Missing values:
  Mean Study Hours in training set = 5.3
  → Fill missing Study Hours in all 3 sets with 5.3

Scaling:
  Fit StandardScaler on 280 training students
  → Apply same scaler to validation and test sets
```

**Step 3 — Cross-validation on training set to tune hyperparameters:**

```
5-Fold CV on 280 training students:

Testing Decision Tree max_depth:
  depth=2:  avg accuracy = 0.74 ± 0.03
  depth=5:  avg accuracy = 0.81 ± 0.02  ← best
  depth=10: avg accuracy = 0.79 ± 0.06  ← higher variance, likely overfitting
```

Choose max_depth=5.

**Step 4 — Retrain final model on full training set, evaluate on validation:**

```
Validation accuracy = 0.82  ← looks good, proceed
```

**Step 5 — Final evaluation on test set (once only):**

```
Test accuracy = 0.80  ← the real number you report
```

The 2% gap between validation (0.82) and test (0.80) is normal. A large gap would signal overfitting to the validation set from too much tuning.

---

## The Data Splitting Checklist

```
Selected Features Received
│
├── 1. Shuffle the data (unless time series)
│
├── 2. Split into Train / Validation / Test
│       - Choose proportions based on dataset size
│       - Lock test set away — do not touch until the very end
│
├── 3. Fit all preprocessing on training set only
│       - Imputation means/medians
│       - Scalers
│       - Encoders
│       → Apply fitted transformers to validation and test sets
│
├── 4. Use validation set (or cross-validation) for tuning
│       - Hyperparameter search
│       - Model comparison
│       - Feature selection validation
│
├── 5. Train final model on full training set
│
├── 6. Evaluate once on test set
│       - Report this number as your model's performance
│       - Do not go back and retune based on this result
│
└── Model ready for deployment
```

---

> **Key takeaway:** Splitting is not just a technical step — it's your contract with reality. The test set is the only honest answer to the question: *does this model actually work?* Treat it like one.
