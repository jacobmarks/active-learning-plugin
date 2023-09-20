import numpy as np

from modAL.models import ActiveLearner
from modAL.multilabel import *
from sklearn.ensemble import RandomForestClassifier
from modAL.batch import uncertainty_batch_sampling

import fiftyone as fo
from fiftyone import ViewField as F


def get_cache():
    g = globals()
    if "_active_learning" not in g:
        g["_active_learning"] = {}

    return g["_active_learning"]


def initialize_learner(
    dataset,
    embeddings_field,
    labels_field,
    batch_size=5,
):
    """Initialize a learner."""
    tagged_samples = dataset.match(F("tags").length() > 0)
    sample_ids = tagged_samples.values("id")

    all_sample_ids = dataset.values("id")

    labeled_view = dataset.select(sample_ids, ordered=True)
    X_init = np.array(labeled_view.values(embeddings_field))
    labeled_ids = labeled_view.values("id")
    unqueried_ids = [id for id in all_sample_ids if id not in labeled_ids]

    labels = labeled_view.values(F("tags")[0])
    unique_labels = sorted(list(set(labels)))
    labels_map = {label: i for i, label in enumerate(unique_labels)}
    y_init = np.array([labels_map[label] for label in labels])

    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_init,
        y_training=y_init,
        query_strategy=uncertainty_batch_sampling,
    )

    cache = get_cache()
    cache["labels_map"] = labels_map
    cache["all_sample_ids"] = all_sample_ids
    cache["learner"] = learner
    cache["embeddings_field"] = embeddings_field
    cache["labels_field"] = labels_field
    cache["unqueried_ids"] = unqueried_ids
    cache["batch_size"] = batch_size

    ## Set the initial predictions
    predict(dataset)


def query_learner(dataset, batch_size=None):
    """Query the learner."""
    cache = get_cache()
    learner = cache["learner"]
    embeddings_field = cache["embeddings_field"]
    unqueried_ids = cache["unqueried_ids"]
    if batch_size is None:
        batch_size = cache["batch_size"]

    unqueried_view = dataset.select(unqueried_ids, ordered=True)
    X_pool = np.array(unqueried_view.values(embeddings_field))
    query_idx, _ = learner.query(X_pool, n_instances=batch_size)

    uvids = unqueried_view.values("id")
    query_ids = [uvids[int(qi)] for qi in query_idx]
    cache["_current_query_ids"] = query_ids
    return query_ids


def _get_label(sample):
    """Get the label of a sample."""
    cache = get_cache()
    labels_map = cache["labels_map"]
    labels_field = cache["labels_field"]

    sample_tags = sample.tags
    if len(sample_tags) == 0:
        label_class = sample[labels_field].label
    else:
        label_class = sample_tags[0]

    return labels_map[label_class]


def teach_learner(dataset):
    """Teach the learner."""
    cache = get_cache()
    learner = cache["learner"]
    embeddings_field = cache["embeddings_field"]
    unqueried_ids = cache["unqueried_ids"]
    query_ids = cache["_current_query_ids"]

    query_view = dataset.select(query_ids, ordered=True)
    X_new = np.array(query_view.values(embeddings_field))
    y_new = np.array([_get_label(sample) for sample in query_view])
    learner.teach(X_new, y_new)
    cache["unqueried_ids"] = [
        id for id in unqueried_ids if id not in query_ids
    ]


def predict(dataset):
    """Predict on the dataset."""
    cache = get_cache()
    learner = cache["learner"]
    embeddings_field = cache["embeddings_field"]
    labels_field = cache["labels_field"]

    X = np.array(dataset.values(embeddings_field))
    y_pred = learner.predict(X)
    y_pred = [list(cache["labels_map"].keys())[i] for i in y_pred]

    vals = [fo.Classification(label=label) for label in y_pred]

    if labels_field not in dataset:
        dataset.add_sample_field(
            labels_field,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification,
        )
    dataset.set_values(labels_field, vals)
    dataset.add_dynamic_sample_fields()
