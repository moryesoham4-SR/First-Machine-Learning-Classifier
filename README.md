# InternCircle - Project 04: First Machine Learning Classifier

This repository contains the complete implementation, Jupyter Notebook, plots, and submission documentation for **InternCircle Data Science & AI Project 04: First Machine Learning Classifier**.

## Project Files
- **[`First_Machine_Learning_Classifier.ipynb`](First_Machine_Learning_Classifier.ipynb)**: Full, interactive Jupyter Notebook with all 8 steps, Markdown annotations, visualizations, and code cells.
- **[`SUBMISSION_WRITEUP.md`](SUBMISSION_WRITEUP.md)**: Formal submission report covering baseline comparison, confusion matrix breakdown, overfitting mechanics, and honest model limitations.
- **[`train_classifier.py`](train_classifier.py)**: Standalone Python script that trains the model and generates all 4 analysis charts.
- **[`titanic.csv`](titanic.csv)**: Local cached copy of the Titanic dataset.
- **Generated Plots:**
  - `depth_vs_accuracy.png`: Plot illustrating tree depth vs. training & test accuracy (demonstrating overfitting).
  - `confusion_matrix.png`: Heatmap breakdown of True Negatives, False Positives, False Negatives, and True Positives.
  - `feature_importances.png`: Horizontal bar chart ranking feature contributions.
  - `decision_tree_structure.png`: Full visual rendering of the fitted Decision Tree (`max_depth=4`).

## Quick Start / Running the Project

### Running in Jupyter / VS Code
Open the directory in VS Code or run:
```bash
jupyter notebook First_Machine_Learning_Classifier.ipynb
```

### Running the Python Script
```bash
python train_classifier.py
```
