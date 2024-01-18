# scikit-learn C4.5 classifier

This is a C4.5 classifier compatible with `scikit-learn`, and more precisely, with `scikit-learn.model_selection.GridSearchCV`.

This repo is forked from [RaczeQ/scikit-learn-C4.5-tree-classifier](https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier), which is in turn based on [zhangchiyu10/pyC45](https://github.com/zhangchiyu10/pyC45).

## Example usage

It is important to pass the feature names to the constructor. In case you use a column transformer, you will need to know the column names beforehand by executing the transformer before the grid search pipeline.

Example usage can be found in a main.py file:

```python
from imblearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA  # for example

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier  # to have another model to check

from c45 import C45

categorical_preprocessing = make_column_transformer(
    ...,
    remainder='passthrough',
    verbose_feature_names_out=False
)

X_tr = categorical_preprocessing.fit_transform(X)
feature_names = categorical_preprocessing.get_feature_names_out()


pipe = Pipeline([
    ('dimensionality_reduction', 'passthrough'),
    ('clf', 'passthrough')
])

param_grid = {
    'dimensionality_reduction': [
        'passthrough',
        PCA()
    ],
    'clf': [
        *[DecisionTreeClassifier(
            random_state=19,
            criterion=prd['criterion'],
            #max_features=prd['max_features'],
            max_depth=prd['max_depth']
        ) for prd in product_dict(
            #max_features=[None, 'sqrt', 'log2'],
            max_depth=[4,5,6,7,8,9,10,None],
            criterion=['gini', 'entropy', 'log_loss'], 
        )],
        C45(attrNames_=feature_names[:-1])
    ]
}

cv = GridSearchCV(pipe, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=10)
cv.fit(X_tr, y)
```
