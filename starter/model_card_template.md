# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a `Random Forest` Model with the following specifications:
n_estimators=200
max_depth=3
random_state=0

## Intended Use
Predicting if the income exceeds 50K.

## Training Data
Extraction was done by Barry Becker from the 1994 Census database.  A set of
reasonably clean records was extracted using the following conditions: 
((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

## Evaluation Data
We evaluate the model on 20% of the data with splits.
The Model itself is built on 80% of the whole data and 20 percent
was used to evaluate the performance.

## Metrics
Precision: 0.9842105263157894
Recall: 0.11724137931034483
fbeta: 0.20952380952380953

## Ethical Considerations
We need to take care of some of the gender and nationality variables and check
the performance on them.

## Caveats and Recommendations
We can try other models such as XGboost and also do hyperparameter optimization.
