# FEATURE SELECTION
# DS -> Data Prep -> Feature Selection

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
│   ├── Feature Selection                          <--- YOU ARE HERE
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

> Feature Selection decides which engineered features to keep and which to discard. More features is not always better — irrelevant features add noise, slow training, and make models harder to interpret.

---

## The 3 Core Concepts

### 1. Why Too Many Features Hurts
> Adding features to a model has diminishing — and eventually negative — returns.

- **Noise** — Irrelevant features introduce random variation. The model wastes capacity fitting to noise instead of signal.
- **Overfitting** — More features means more ways to memorize the training data rather than learn generalizable patterns.
- **Curse of Dimensionality** — In high-dimensional space, data points become sparse. Distance-based models (k-NN, SVM) stop working well because everything becomes equally "far" from everything else.
- **Multicollinearity** — Highly correlated features carry redundant information and destabilize regression coefficients.

```
10 features, 2 useful:   model spends most effort fitting noise
10 features, 9 useful:   model is efficient, accurate
100 features, 9 useful:  model overfits, slow, hard to interpret
```

---

### 2. Redundancy vs. Irrelevance
> Two different reasons to drop a feature.

- **Irrelevant Feature** — Has no relationship with the target. Example: House_ID when predicting price.
- **Redundant Feature** — Carries the same information as another feature already in the model. Example: both `Size_sqft` and `Size_sqm` (they're the same measurement).

Both should be removed — but for different reasons and by different methods.

---

### 3. The Three Approaches
> Feature selection methods differ in how they decide which features to keep.

| Approach | Uses the model? | Speed | Best for |
|---|---|---|---|
| **Filter** | No | Fast | Quick first pass, large datasets |
| **Wrapper** | Yes | Slow | Best possible subset, smaller datasets |
| **Embedded** | Yes (during training) | Medium | Efficient — selection and training in one step |

---

## Selection Methods

### Filter Methods
> Score each feature independently based on its statistical relationship with the target, then keep the top-ranked ones.

**When to use:** A fast first pass to eliminate obviously useless features before anything more expensive. No model training required.

**Common metrics:**

| Method | Target type | What it measures |
|---|---|---|
| **Pearson Correlation** | Continuous (regression) | Linear relationship between feature and target |
| **Chi-Squared (χ²)** | Categorical (classification) | Statistical dependence between categorical feature and target |
| **ANOVA F-test** | Continuous feature, categorical target | Variance between class means vs. within-class variance |
| **Variance Threshold** | Any | Drops features where almost all values are the same |

**Example — Pearson Correlation filter for house price prediction:**

| Feature      | Correlation with Price | Keep? |
|--------------|------------------------|-------|
| Size         | 0.85                   | ✅    |
| Bedrooms     | 0.72                   | ✅    |
| Bathrooms    | 0.68                   | ✅    |
| Neighborhood | 0.54                   | ✅    |
| LotAge       | 0.08                   | ❌    |
| House_ID     | 0.01                   | ❌    |

LotAge and House_ID have near-zero correlation with price — drop them before training.

**Variance Threshold example:**
```
Feature: Has_Pool
  House A → 0
  House B → 0
  House C → 0
  House D → 0
  House E → 1

Variance ≈ 0.04 → nearly constant → carries almost no information → drop
```

> **Limitation:** Filter methods evaluate features one at a time and miss interaction effects. A feature that looks weak alone might be powerful in combination with another.

---

### Wrapper Methods
> Use the model itself to evaluate subsets of features — train on a subset, score it, adjust, repeat.

**When to use:** You want the best possible feature subset and can afford the computational cost. Works well on small-to-medium datasets.

**Three common approaches:**

**Forward Selection:** Start with no features. Add one at a time, keeping whichever improves the model most at each step.

```
Start:    {} → baseline score
Step 1:   {Size} → R² = 0.72          ← best single feature
Step 2:   {Size, Bedrooms} → R² = 0.81
Step 3:   {Size, Bedrooms, Neighborhood} → R² = 0.87
Step 4:   {Size, Bedrooms, Neighborhood, LotAge} → R² = 0.87  ← no gain, stop
```

**Backward Elimination:** Start with all features. Remove the least useful one at a time.

```
Round 1: All 6 features → R² = 0.88
  Remove House_ID → R² = 0.88  ✅ no loss

Round 2: 5 features → R² = 0.88
  Remove LotAge → R² = 0.87   ✅ negligible loss

Round 3: 4 features → R² = 0.87
  Remove Bathrooms → R² = 0.83  ❌ meaningful drop → stop
```

Final selected features: Size, Bedrooms, Neighborhood, Bathrooms.

**Recursive Feature Elimination (RFE):** Train the model on all features, rank them by importance, remove the weakest N, retrain. Repeat until the target number of features is reached.

```
Iteration 1: Train on 6 features → rank by coefficient size → remove House_ID
Iteration 2: Train on 5 features → rank → remove LotAge
Iteration 3: Train on 4 features → rank → performance drops → stop
```

| | Forward Selection | Backward Elimination | RFE |
|---|---|---|---|
| **Starts with** | No features | All features | All features |
| **Direction** | Adds | Removes | Removes in batches |
| **Best for** | Many features (faster start) | Fewer features | When you know target feature count |
| **Computationally** | Moderate | Moderate | Efficient |

> **Limitation:** Wrapper methods are expensive — evaluating every possible subset of N features requires 2ᴺ model trainings in the brute-force case. Forward/backward heuristics reduce this but may miss the true optimal subset.

---

### Embedded Methods
> Feature selection happens automatically as part of model training. The model learns which features matter while it learns to predict.

**When to use:** You're already using a model that supports it (Lasso, Random Forest, XGBoost). The most efficient approach — no separate selection step needed.

**Lasso Regression (L1 Regularization):**

Lasso adds a penalty for large coefficients that drives irrelevant features exactly to zero during training.

```
Standard Linear Regression:
  Price = 50k + 150(Size) + 9k(Bedrooms) + 800(LotAge) + 200(House_ID)

After Lasso:
  Price = 50k + 148(Size) + 8.5k(Bedrooms) + 0(LotAge) + 0(House_ID)
                                              ↑ eliminated  ↑ eliminated
```

LotAge and House_ID contribute nothing — Lasso zeros them out automatically.

**Random Forest Feature Importance:**

After training, each feature is ranked by how much it reduced impurity (Gini or entropy) across all trees and all splits.

| Feature      | Importance Score |
|--------------|-----------------|
| Size         | 0.42            |
| Neighborhood | 0.28            |
| Bedrooms     | 0.19            |
| Bathrooms    | 0.08            |
| LotAge       | 0.02            |
| House_ID     | 0.01            |

Drop LotAge and House_ID — they contribute almost nothing across the entire forest.

**XGBoost Feature Importance:**

XGBoost tracks three importance metrics:

| Metric | What it measures |
|---|---|
| **Weight** | Number of times a feature was used in a split |
| **Gain** | Average improvement in accuracy when the feature is used |
| **Cover** | Average number of samples affected by splits on this feature |

Gain is usually the most informative — a feature used rarely but with high gain is more valuable than one used often with low gain.

```
Feature importances (by Gain):
  Size         → 0.61  ✅ keep
  Neighborhood → 0.22  ✅ keep
  Bedrooms     → 0.12  ✅ keep
  Bathrooms    → 0.04  ⚠️ borderline
  LotAge       → 0.01  ❌ drop
  House_ID     → 0.00  ❌ drop
```

> **Key advantage of Embedded methods:** No extra computation — you get feature importances for free as a byproduct of training the model you were going to train anyway.

---

## The Feature Selection Checklist

```
Engineered Data Received
│
├── 1. Quick filter pass
│       - Drop near-zero variance features
│       - Drop features with correlation < threshold to target
│       - Drop one of any pair of features with correlation > 0.95 (redundant)
│
├── 2. Choose a selection method
│       - Large dataset, need speed     → Filter Methods
│       - Small dataset, need best      → Wrapper (RFE or Backward Elimination)
│       - Using Lasso / RF / XGBoost    → Embedded (free, use it)
│
├── 3. Validate on held-out data
│       - Does removing these features hurt validation performance?
│       - If yes → reconsider, some may carry signal
│
├── 4. Document what was dropped and why
│
└── Selected features → ready for Data Splitting & Model Training
```

---

> **Key takeaway:** Feature Selection is not about removing features you don't like — it's about removing features the model can't use. The signal-to-noise ratio of your feature set matters as much as the algorithm you choose.
