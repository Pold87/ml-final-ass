# Material for the Course: Machine Learning (IN4320)

The file `R/imputation_final.R` conducts the imputation with the
method Multiple Imputation by Chained Equations (MICE). This yields
five training and test sets with imputed values. The evaluation of the missing values with Little's MCAR test is in `R/missing_eval.R`.

The code for the creation of the dummy variables is in `Python/onehot.py`

The file `Python/cross_validation.py`, contains the code for the
cross-validation and for creating preliminary submissions. The final submissions is created using the file `Python/combine.py`.