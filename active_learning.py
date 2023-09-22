import numpy as np

from modAL.models import ActiveLearner
from modAL.multilabel import *
from modAL.batch import uncertainty_batch_sampling

import fiftyone as fo
from fiftyone import ViewField as F


def get_cache():
    g = globals()
    if "_active_learning" not in g:
        g["_active_learning"] = {}

    return g["_active_learning"]


def _create_estimator(ctx):
    estimator = ctx.params["estimator"]
    if estimator == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        estimator = KNeighborsClassifier()
    elif estimator == "svc":
        from sklearn.svm import SVC

        estimator = SVC()
    elif estimator == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier()
    return estimator


def _create_random_forest_classifier(ctx):
    """Create a random forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    n_estimators = ctx.params.get("n_estimators", 100)
    max_depth = ctx.params.get("max_depth", None)
    min_samples_split = ctx.params.get("min_samples_split", 2)
    min_samples_leaf = ctx.params.get("min_samples_leaf", 1)
    criterion = ctx.params.get("criterion", "gini")

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
    )


def _create_gradient_boosting_classifier(ctx):
    """Create a gradient boosting classifier."""
    from sklearn.ensemble import GradientBoostingClassifier

    n_estimators = ctx.params["n_estimators"]
    # max_depth = ctx.params["max_depth"]
    max_depth = ctx.params.get("max_depth", 3)
    learning_rate = ctx.params.get("learning_rate", 0.1)
    subsample = ctx.params.get("subsample", 1.0)
    min_samples_split = ctx.params.get("min_samples_split", 2)
    min_samples_leaf = ctx.params.get("min_samples_leaf", 1)
    max_leaf_nodes = ctx.params.get("max_leaf_nodes", None)

    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
    )


def _create_bagging_classifier(ctx):
    """Create a bagging classifier."""
    from sklearn.ensemble import BaggingClassifier

    estimator = _create_estimator(ctx)
    n_estimators = ctx.params["n_estimators"]

    max_samples = ctx.params.get("max_samples", 1.0)
    max_features = ctx.params.get("max_features", 1.0)
    bootstrap = ctx.params.get("bootstrap", True)
    bootstrap_features = ctx.params.get("bootstrap_features", False)

    return BaggingClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        bootstrap_features=bootstrap_features,
    )


def _create_adaboost_classifier(ctx):
    """Create an AdaBoost classifier."""
    from sklearn.ensemble import AdaBoostClassifier

    estimator = _create_estimator(ctx)
    n_estimators = ctx.params.get("n_estimators", 50)
    learning_rate = ctx.params.get("learning_rate", 1.0)

    return AdaBoostClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )


def _create_classifier(ctx):
    """Create a classifier."""
    classifier_type = ctx.params.get("classifier_type", "Random Forest")
    if classifier_type == "Random Forest":
        return _create_random_forest_classifier(ctx)
    elif classifier_type == "Gradient Boosting":
        return _create_gradient_boosting_classifier(ctx)
    elif classifier_type == "Bagging":
        return _create_bagging_classifier(ctx)
    elif classifier_type == "AdaBoost":
        return _create_adaboost_classifier(ctx)
    else:
        raise ValueError("Unknown classifier type '%s'" % classifier_type)


def initialize_learner(ctx):
    """Initialize a learner."""

    classifier = _create_classifier(ctx)
    embeddings_field = ctx.params["embeddings_field"]
    labels_field = ctx.params["labels_field"]
    batch_size = ctx.params["batch_size"]

    dataset = ctx.dataset
    all_sample_ids = dataset.values("id")

    if ctx.params.get("init_labels", "labels") == "tags":
        ## Use tags as the initial labels
        ### Assume tags are correct and mutually exclusive
        tagged_samples = dataset.match(F("tags").length() > 0)
        sample_ids = tagged_samples.values("id")
        labeled_view = dataset.select(sample_ids, ordered=True)
        X_init = np.array(labeled_view.values(embeddings_field))
        labeled_ids = labeled_view.values("id")
        unqueried_ids = [id for id in all_sample_ids if id not in labeled_ids]
        labels = labeled_view.values(F("tags")[0])
    else:
        ## Use labels as the initial labels
        ### Don't assume labels are correct
        init_label_field = ctx.params["init_label_field"]
        labeled_view = dataset.match(F(f"{init_label_field}.label"))
        X_init = np.array(labeled_view.values(embeddings_field))
        unqueried_ids = all_sample_ids
        labels = labeled_view.values(f"{init_label_field}.label")

    unique_labels = sorted(list(set(labels)))
    labels_map = {label: i for i, label in enumerate(unique_labels)}
    y_init = np.array([labels_map[label] for label in labels])

    learner = ActiveLearner(
        estimator=classifier,
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
