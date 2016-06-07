X = csvread('../data/train.csv', 1);

is_cat = logical([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]);

X_cat = X(:, is_cat);

% I used 2 principal components

