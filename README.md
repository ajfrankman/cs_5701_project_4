# CS 5701 Project 4: Decision Tree Accuracy Study

This project implements a small decision tree classifier for binary classification tasks and evaluates its performance on several CSV datasets. The program repeatedly trains a tree using different training ratios, then records the average training and testing accuracies for each dataset.

## Project Goal

The goal of this assignment is to:

- parse tabular CSV data into a set of binary features and a target label,
- build a decision tree using information gain,
- randomly split each dataset into training and testing subsets,
- repeat the experiment across multiple runs and training ratios,
- save the averaged results to a CSV report.

## Files

- `main.py` — entry point for the experiment; defines the datasets and runs the evaluation pipeline.
- `decision_tree.py` — contains the `DecisionTree` implementation, including entropy, information gain, tree construction, and prediction logic.
- `helpers.py` — parsing and repeated-case evaluation helpers.
- `data/` — input datasets used by the program.
- `results.csv` — generated summary of average training and testing accuracy by dataset and training ratio.

## Datasets

The project evaluates the following files in the `data/` directory:

- `Problem4DecisionTreeData.csv`
- `minus_1_col.csv`
- `minus_2_col.csv`

Each CSV contains the following structure:

- an example index column,
- multiple binary feature columns,
- a final target column where the decision is either `Yes` or `No`.

## How to Run

From the project root, execute:

```bash
python3 main.py
```

This will:

1. load each dataset,
2. run the decision tree experiment for training ratios from `0.1` to `0.9`,
3. repeat each experiment `100` times,
4. write the aggregate accuracy table to `results.csv`.

## Output

The generated `results.csv` file contains a summary table with columns for:

- training ratio,
- training accuracy,
- testing accuracy,

for each dataset.

## Notes

- The tree uses binary feature values and selects the attribute with the highest information gain at each split.
- The implementation uses randomized train/test splits for each run, which makes the results suitable for measuring average generalization performance.
- The current repository includes a sample `results.csv` generated from the experiment runs.

