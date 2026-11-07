# InternCircle Project 04: First Machine Learning Classifier
**Author:** InternCircle Learner  
**Domain:** Data Science & AI  
**Project:** Supervised Machine Learning Classifier on the Titanic Dataset  
**Tools Used:** Python 3.12, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn  

---

## 1. Problem & Prediction Definition

> **"Predict whether a passenger survived from their class, sex, age and fare. Survived is the target; the rest are features."**

* **Target Variable ($y$):** `survived` (Binary classification: `0` = Did not survive, `1` = Survived).
* **Feature Variables ($X$):**
  * `pclass`: Socioeconomic status / ticket class ($1^{\text{st}}$, $2^{\text{nd}}$, $3^{\text{rd}}$).
  * `sex`: Passenger biological sex (encoded numerically: Female = $1$, Male = $0$).
  * `age`: Passenger age in fractional/integer years.
  * `fare`: Passenger ticket fare in British Pounds.

---

## 2. Data Preparation & Splitting

1. **Handling Missing Values:**
   * Raw dataset size: **891 rows**.
   * Filtering to the five required attributes (`survived`, `pclass`, `sex`, `age`, `fare`) and removing rows containing missing values (`dropna()`) yields **714 clean rows** (177 rows removed, overwhelmingly due to unrecorded ages).
2. **Feature Encoding:**
   * Transformed categorical `sex` into an integer indicator: `(df["sex"] == "female").astype(int)`.
3. **Stratified Train/Test Split:**
   * Train/Test ratio: **80% training** ($571$ passengers) and **20% testing** ($143$ passengers).
   * Fixed `random_state=42` to guarantee exact computational reproducibility.
   * Parameter `stratify=y` enforced identical class distributions across both splits:
     * Train survival rate: **40.6%** ($232$ survivors, $339$ non-survivors).
     * Test survival rate: **40.6%** ($58$ survivors, $85$ non-survivors).

---

## 3. Naive Baseline Comparison

To determine whether the machine learning classifier provides authentic predictive utility, we set a heuristic baseline using the majority class rule:
* In the test set of 143 passengers, $85$ died ($59.4\%$) and $58$ survived ($40.6\%$).
* A naive model that simply guesses the most frequent class ("Did not survive" = 0) for every single passenger achieves:
  $$\text{Baseline Accuracy} = \max(\bar{y}_{\text{test}}, 1 - \bar{y}_{\text{test}}) = 1 - 0.4056 = \mathbf{0.594 \ (59.4\%)}$$
* **Threshold of Value:** Any valid machine learning model must comfortably exceed $59.4\%$ test accuracy to justify its deployment.

---

## 4. Model Training & Test Set Evaluation

We trained a Scikit-Learn `DecisionTreeClassifier` with `max_depth=4` and `random_state=42` strictly on the training partition (`X_train`, `y_train`).

### Performance Metrics Summary

| Metric | Baseline | Model (Train Set) | Model (Test Set) |
|---|:---:|:---:|:---:|
| **Accuracy** | $59.4\%$ | $84.8\%$ | **$79.7\%$** |
| **Gain vs. Baseline** | --- | $+25.4\%$ | **$+20.3\%$** |

The trained decision tree correctly classifies **114 out of 143 test passengers**, outperforming the naive baseline by **over 20 percentage points**.

---

## 5. Confusion Matrix Analysis

Evaluating accuracy alone can be misleading in imbalanced settings. We examine the full test set confusion matrix:

$$\begin{pmatrix} 72 & 13 \\ 16 & 42 \end{pmatrix}$$

```
                       PREDICTED: Died (0)     PREDICTED: Survived (1)
 ACTUAL: Died (0)             72 (TN)                   13 (FP)
 ACTUAL: Survived (1)         16 (FN)                   42 (TP)
```

### Detailed Cell-by-Cell Breakdown:
1. **True Negatives ($\text{TN} = 72$):**
   * Passengers who actually did not survive and were correctly classified as deceased by the model.
2. **False Positives ($\text{FP} = 13$ - Type I Error):**
   * Passengers who actually perished, but the model incorrectly predicted they survived.
3. **False Negatives ($\text{FN} = 16$ - Type II Error):**
   * Passengers who actually survived, but the model mistakenly predicted they died.
4. **True Positives ($\text{TP} = 42$):**
   * Passengers who actually survived and were correctly identified as survivors by the model.

### Derived Diagnostic Rates:
* **Precision (Survived):** $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{42}{42 + 13} = \mathbf{76.4\%}$ (When the model predicts survival, it is correct $76.4\%$ of the time).
* **Recall / Sensitivity (Survived):** $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{42}{42 + 16} = \mathbf{72.4\%}$ (The model captures $72.4\%$ of all actual survivors).
* **Specificity (Died):** $\frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{72}{72 + 13} = \mathbf{84.7\%}$.

---

## 6. Feature Importances

Using Gini impurity reduction across tree splits (`model.feature_importances_`), we obtain the relative predictive contributions:

| Feature | Gini Importance | Historical & Domain Rationale |
|---|:---:|---|
| **`sex`** | **$0.55 \ (55\%)$** | **Dominant factor.** The maritime evacuation strictly enforced the protocol of *"women and children first"*. Female passengers had vastly higher lifeboat boarding rates. |
| **`pclass`** | **$0.24 \ (24\%)$** | **Second most critical factor.** First and second class cabins were situated on the upper decks with direct access to lifeboats; steerage ($3^{\text{rd}}$ class) passengers were positioned deeper below deck with delayed access. |
| **`age`** | **$0.14 \ (14\%)$** | Young children received evacuation preference over adult males across all ticket tiers. |
| **`fare`** | **$0.08 \ (8\%)$** | Correlates strongly with ticket class and physical cabin location aboard the vessel. |

---

## 7. Tree Depth and Overfitting Analysis

To empirically demonstrate the danger of overfitting, we varied `max_depth` from $1$ to $12$ while tracking both training and test set accuracy:

| Depth | Training Accuracy | Test Accuracy | Overfitting Gap ($\text{Train} - \text{Test}$) | Diagnostic State |
|:---:|:---:|:---:|:---:|---|
| **1** | $77.9\%$ | $78.3\%$ | $-0.4\%$ | Underfitting (decision stump on `sex`) |
| **2** | $79.5\%$ | $77.6\%$ | $+1.9\%$ | Mild underfitting |
| **3** | $81.8\%$ | $\mathbf{81.1\%}$ | $+0.7\%$ | Optimal generalization |
| **4** | $84.8\%$ | **$79.7\%$** | $+5.1\%$ | Selected model (balanced, interpretable) |
| **5** | $86.5\%$ | $76.2\%$ | $+10.3\%$ | Onset of overfitting |
| **6** | $88.3\%$ | $78.3\%$ | $+10.0\%$ | Variance increasing |
| **7** | $91.6\%$ | $78.3\%$ | $+13.3\%$ | Memorizing local noise |
| **8** | $92.5\%$ | $79.0\%$ | $+13.5\%$ | Deep branches splitting single instances |
| **9** | $94.7\%$ | $76.9\%$ | $+17.8\%$ | Severe overfitting |
| **10** | $95.6\%$ | $76.2\%$ | $+19.4\%$ | Severe overfitting |
| **11** | $96.5\%$ | $76.9\%$ | $+19.6\%$ | Near-memorization of training samples |
| **12** | $96.8\%$ | $77.6\%$ | $+19.2\%$ | Pathological overfitting |

### Why Very Deep Trees Perform Worse on New Data:
1. **Splitting on Noise Instead of Signal:** As tree depth increases beyond 4, internal decision rules partition nodes into tiny subsets (often leaves containing only 1 or 2 passengers). The tree fits idiosyncratic anomalies (e.g. an unusual survival of an elderly 3rd class male) rather than generalizable trends.
2. **High Model Variance:** Unconstrained trees suffer from high variance and near-zero bias. While training accuracy climbs toward $100\%$, test accuracy plateaus and declines, creating an overfitting gap exceeding $19\%$.
3. **Importance of Regularization:** Constraining depth to $3$ or $4$ acts as implicit regularization, preventing the model from chasing noise and keeping decision rules robust and interpretable.

---

## 8. Honest Limitations of the Model

1. **Information Loss from Missing Value Deletion (`dropna`):**
   * Removing rows with missing data discarded $177$ passengers ($19.9\%$ of the dataset), solely because their age was unrecorded.
   * Missing ages were disproportionately common in third-class passengers. Dropping them introduces potential survivor selection bias that could skew survival probability estimates.
2. **Small Dataset Sensitivity:**
   * With a test set of only $143$ instances, each individual passenger prediction shifts the test accuracy metric by $\approx 0.7\%$. Small datasets yield relatively noisy metric estimates; cross-validation (e.g., 5-fold CV) would provide narrower confidence intervals.
3. **Instability of Single Decision Trees:**
   * Individual decision trees are high-variance estimators that are notoriously sensitive to minor perturbations in the training split. Small variations in sample distribution can alter the root split thresholds.
4. **Omission of High-Signal Contextual Features:**
   * By restricting features strictly to `pclass`, `sex`, `age`, and `fare`, the model ignores family dynamics (`sibsp` and `parch`), title honorifics (e.g., 'Master' for boys vs. 'Mr' for men), and deck location extracted from cabin numbers.
5. **Recommended Path to Production:**
   * Imputing missing ages (e.g., via median age grouped by title and passenger class) and deploying an ensemble classifier (such as Random Forest or XGBoost) would mitigate tree variance while lifting test accuracy above $82\%-84\%$.

---

## 9. Verification Checklist Compliance

- [x] **The target and features are defined in a sentence**
- [x] **Training and test sets are separate**
- [x] **A baseline is calculated and compared ($59.4\%$ vs $79.7\%$)**
- [x] **Accuracy and all 4 cells of the confusion matrix are explained**
- [x] **Overfitting is shown with different tree depths ($1$ to $12$ table and plot)**
- [x] **Limitations of the model are stated honestly**
