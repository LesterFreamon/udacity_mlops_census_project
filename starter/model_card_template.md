# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a LogisticRegression trained to predict whether a person makes over 50K a year based on features like their education, marital status, occupation, etc. The model was trained with hyperparameter tuning to optimize accuracy score.

## Intended Use
The intended use of this model is to help understand factors that contribute to income levels and potentially to help with job placement or career counseling. It should not be used to make decisions about hiring or compensation.

## Training Data
The model was trained on the UCI Adult dataset, which includes about 32k instances. Each instance represents an individual and includes features like age, workclass, education, marital status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. The label is whether the individual makes more than 50k a year.

## Evaluation Data
The evaluation data is a subset of the UCI Adult dataset, comprising 20% of the original data. This set was kept separate from the training data and used solely for evaluation purposes.

## Metrics
### General

The model is evaluated based on precision, recall, and fbeta score.

| Key       | Value   |
|-----------|---------|
| precision | 0.7422    |
| recall    | 0.6065   |
| fbeta     | 0.6675 |

### Slicing for education

| education | Precision | Recall | Fbeta |
|---|---|---|---|
| 5th_6th | 1.0 | 0.2 | 0.3333 |
| assoc_voc | 0.6964 | 0.5820 | 0.6341 |
| some_college | 0.63483146 | 0.42481203 | 0.509009 |
| hs_grad | 0.7086 | 0.2662 | 0.3870 |
| masters | 0.7644 | 0.8865 | 0.8210 |
| 11th | 1.0 | 0.0909 | 0.1666 |
| bachelors | 0.7550 | 0.8309 | 0.7911 |
| prof_school | 0.8591 | 0.8472 | 0.8531 |
| 7th_8th | 1.0 | 0.0 | 0.0 |
| assoc_acdm | 0.7 | 0.5555 | 0.6194 |
| 9th | 0.0 | 0.0 | 0.0 |
| 1st_4th | 1.0 | 0.0 | 0.0 |
| 10th | 0.5 | 0.05 | 0.0909 |
| doctorate | 0.8769 | 0.9344 | 0.9047 |
| 12th | 1.0 | 0.1818 | 0.3076 |
| preschool | 0.0 | 1.0 | 0.0 |


## Ethical Considerations
While the model is intended to help understand factors contributing to income levels, it should be noted that the model's predictions could potentially be influenced by biases present in the training data. For instance, if certain occupations are dominated by one gender, the model may reflect this bias in its predictions. Therefore, results should be interpreted with caution, keeping in mind the societal context and potential biases of the data.

## Caveats and Recommendations
The model assumes that the distribution of features and the relationships between features and income have not changed since the data was collected in 1996. If these factors have changed, the model's predictions may not be accurate. Future work could involve training the model on more recent data, if available, or re-evaluating the model's performance periodically to ensure it remains accurate. Users of the model should keep these factors in mind when interpreting the model's predictions.
