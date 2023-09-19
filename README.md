## Active Learning

This plugin is a Python plugin that allows you to label your dataset with
active learning, using the
[modAL](https://modal-python.readthedocs.io/en/latest/) library.

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/active-learning-plugin
```

Then install the requirements:

```shell
pip install -r requirements.txt
```

## Operators

### `create_active_learner`

Creates an active learner that can be used to label your dataset. It is initialized
by taking the tags on your dataset as the classes to be labeled.

You select the embeddings field to use for the active learning. The embeddings
will be the features used to train the active learner.

Additionally, you name the predictions field that will be created by the active
learner. This field will contain the predictions of the active learner on the
embeddings field, and will be used to select the next samples to label.

You can also set the number of samples to label at each iteration.

### `query_learner`

Queries the active learner for the next samples to label. Tag the samples whose
predicted labels are incorrect. Untagged samples will be treated as correct
predictions.

### `update_learner_predictions`

The active learner will be retrained on the newly labeled
samples, and the predictions field will be updated.
