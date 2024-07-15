![logo](logo.jpg)

[![(License MIT 1.0)](https://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

[license]: LICENSE

**Disclaimer: This is just a fun experiment, I conducted for my curiosity and entertainment. It's not intended to be
useful for anything else.**

# Yo-dawg-ml-model architecture

![yo_dawg](yo_dawg.jpg)

The Yo-dawg-ml-model architecture (also known as the Yung-Kofman-Hermann architecture) is a generalization of the
classical decision trees.
Instead of the decision nodes splitting the data on a simple feature threshold, arbitrary model types (linear
regression, support-vector machine, multi-layer perceptron) are put in place here.
In addition, the leaf nodes also don't have a fixed value, and sub-models do the prediction part here too.

## Is this helpful?

Maybe. Not sure yet if the additional complexity and computational cost are worth it.

## Implementation

The basis is a simplified standard recursive decision-tree implementation using the Gini impurity to select a good
decider on each node. It can only handle numerical features and only supports regression (not classification).
It gives similar results to the scikit-learn implementation:

```
sklearn.tree train RMSE: 235073.99796858433
sklearn.tree test RMSE: 1772486.2439492478
custom.tree train RMSE: 235073.99796858433
custom.tree test RMSE: 1771910.6788249828
```

With the advanced deciders enabled, it still works,
and sometimes gives better results on the test set than the "boring" version (cherry-picked result):

```
yo-dawg.tree train RMSE: 235073.99796858433
yo-dawg.tree test RMSE: 1733936.984241994
```

## Origin story

Our team was working on predicting specific scores for images.
At one point, we tested something like this:

```python3
score = model_a.predict(image)
if score > 0.8:
    score = model_b.predict(image)
```

**Obviously** this is a small decision tree,
with decision thresholds applied to a model score instead of a simple input feature.
So this repository is the logical consequence of expanding on this idea.
