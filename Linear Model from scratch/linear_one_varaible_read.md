# Linear Regression with One Variable: Gradient Descent from Scratch

A from-scratch implementation of linear regression with one feature, built while studying **Course 1, Week 1** of Andrew Ng's *Machine Learning Specialization*. The cost function, gradient, and gradient descent are written in plain Python/NumPy and then verified against scikit-learn on synthetic and real data.

## What this project covers

- Linear regression model `f(x) = wx + b`
- Mean squared error cost function
- Gradient descent (batch, simultaneous update of `w` and `b`) with early stopping and divergence detection
- Verification on a hand-checkable dataset, then on synthetic data with a known true line
- Application to a real Experience-Salary dataset
- A study of how the learning rate (alpha) behaves on raw, unscaled data

## Workflow

1. Implement `compute_cost`, `compute_gradient`, and `gradient_descent`
2. Sanity-check them on two points that lie exactly on `y = 200x + 100`
3. **Experiment 1:** synthetic data `y = 3x + 7 + noise`, verified with scikit-learn
4. **Experiment 2:** real Experience-Salary data (random sample of 100 rows)
5. **Experiment 3:** learning-rate study on the raw data
6. Visualize the best-fit line and the cost curve

## Results

| Experiment | My gradient descent | scikit-learn | Final cost |
|---|---|---|---|
| Synthetic (`y = 3x + 7 + noise`) | w = 2.91, b = 7.43 | w = 2.91, b = 7.43 | 1.61 |
| Real data (Experience vs. Salary) | w = 0.75, b = 6.91 | w = 0.75, b = 6.91 | 12.17 |

My implementation matches scikit-learn in both cases, so the algorithm is correct. The learned synthetic values are close to the true `w = 3`, `b = 7` but not identical, because the best line for 100 noisy points differs slightly from the hidden true line.

### Learning rate on raw (unscaled) data

| alpha | Behavior |
|---|---|
| 0.1, 0.01 | Diverged: steps overshoot the minimum and the cost blows up to `inf` / `nan` |
| 0.001 | Converged fastest (best of the values tested) |
| 0.0001 | Stable but slow; `b` is still far from its final value after many iterations |

## Key learnings

1. **The learning rate depends on the scale of the features.** With small or moderate feature values a moderate alpha works. With large values the gradients are large, so alpha must be very small, and then convergence is slow. This is why feature scaling is needed. What matters is the size of the values, not the number of rows.
2. **Model performance depends on data quality.** If the data is scattered and noisy, even the best-fit line leaves large errors. On the salary data, gradient descent and scikit-learn gave the same answer, so the limit came from the data, not the algorithm.
3. **Real-world outcomes depend on multiple features.** Salary depends on more than experience (role, skills, education, company). A one-feature model treats all of that as noise, which motivates multiple linear regression.

Other takeaways: test functions on tiny inputs with known answers, plot cost against iterations (it should go down on every iteration with a good alpha), and always verify against a trusted library.

## Repository structure

```
.
├── linear_regression_gradient_descent.ipynb   # main notebook

 Experience-Salary.csv                   # not included, see below
├── 
└── README.md
```

## How to run

```bash

jupyter notebook linear_regression_gradient_descent.ipynb
```

Experiment 2 needs the dataset at `data/Experience-Salary.csv` with the columns `exp(in months)` and `salary(in thousands)`. Experiment 1 (synthetic data) runs without any file.

## Next steps

- Multiple linear regression (several features)
- Feature scaling, then repeat the learning-rate study and compare the workable alpha range
- Vectorized NumPy implementation instead of Python loops

## Acknowledgements

Course: [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) by Andrew Ng (DeepLearning.AI and Stanford Online).

## MY Profile

Muhammad Aqib · [GitHub](https://github.com/khanaqibkhan378-ui) · [LinkedIn](https://www.linkedin.com/in/muhammadaqib-uos06)
