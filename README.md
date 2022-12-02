# Polynomial Expansion

Implementation of a deep learning model that learns to expand single variable polynomials, where the model takes the factorized sequence as input and predict the expanded sequence. This is an exercise to demonstrate your machine learning prowess, so please refrain from parsing or rule-based methods.

A training file is provided in S3:
* `train.txt` : https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt

Each line of `train.txt` is an example, the model should take the factorized form as input, and predict the expanded form. E.g.

* `n*(n-11)=n**2-11*n`
* `n*(n-11)` is the factorized input
* `n**2-11*n`  is the expanded target

The expanded expressions are commutable, but only the form provided is considered as correct.
