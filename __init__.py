"""Active Learning plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
import numpy as np
import os

from bson import json_util

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F


def _is_teams_deployment():
    val = os.environ.get("FIFTYONE_INTERNAL_SERVICE", "")
    return val.lower() in ("true", "1")


TEAMS_DEPLOYMENT = _is_teams_deployment()

if not TEAMS_DEPLOYMENT:
    with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
        # pylint: disable=no-name-in-module,import-error
        from active_learning import (
            initialize_learner,
            query_learner,
            teach_learner,
            predict,
        )


def get_vector_fields(dataset):
    """Get all vector fields in a dataset."""
    vector_fields = []
    fields = dataset.get_field_schema(flat=True)
    for field, ftype in fields.items():
        full_type = str(ftype)
        if "VectorField" in full_type:
            vector_fields.append(field)
    return vector_fields


def _get_candidate_feature_fields(dataset):
    sample = dataset.first()
    fields = []
    for fn in sample.field_names:
        if sample[fn].__class__ in [float, np.ndarray]:
            fields.append(fn)
    return fields


def _ensure_feature_fields(cand_feature_fields, inputs):
    if len(cand_feature_fields) == 0:
        inputs.view(
            "warning",
            types.Warning(
                label="No Feature Fields",
                description=(
                    "You must have float fields or vector fields on your dataset."
                    " These will be used as input to the model. You can create"
                    " a vector field by running `dataset.compute_embeddings()`."
                ),
            ),
        )
        return False
    return True


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _get_classification_fields(dataset):
    classif_fields = []
    fields = dataset.get_field_schema()
    for field_name, field_type in fields.items():
        if "Classification" in str(field_type):
            classif_fields.append(field_name)
    return classif_fields


def _ensure_tags(dataset, inputs):
    unique_tags = dataset.distinct("tags")
    if len(unique_tags) == 0:
        inputs.view(
            "warning",
            types.Warning(
                label="No Tags",
                description=(
                    "You must have at least two distinct tags on your dataset."
                ),
            ),
        )
        return False
    elif len(unique_tags) == 1:
        inputs.view(
            "warning",
            types.Warning(
                label="Only One Tag",
                description=(
                    "To use active learning, you must have at least two tags on your dataset."
                    " These are used as initial labels."
                ),
            ),
        )
        return False
    return True


def _initial_labels_selection(inputs):
    init_label_choices = types.RadioGroup()
    init_label_choices.add_choice(
        "labels",
        label="Label field",
        description="Select a field containing labels to use as initial labels",
    )
    init_label_choices.add_choice(
        "tags",
        label="Tags",
        description="Use tags as initial labels",
    )
    inputs.enum(
        "init_labels",
        init_label_choices.values(),
        view=init_label_choices,
        default="labels",
    )


def _configure_random_forest_classifier(inputs):
    inputs.view(
        "rf_header",
        types.Header(
            label="Random Forest Classifier",
            description="Configure the Random Forest Classifier",
            divider=True,
        ),
    )
    inputs.int("n_estimators", label="Number of estimators", default=100)
    inputs.int("max_depth", label="Max depth", default=None)
    inputs.int("min_samples_split", label="Min samples split", default=2)
    inputs.int("min_samples_leaf", label="Min samples leaf", default=1)

    criteria_choices = ["gini", "entropy", "log_loss"]
    criteria_group = types.RadioGroup()
    for choice in criteria_choices:
        criteria_group.add_choice(choice, label=choice)

    inputs.enum(
        "criteria",
        criteria_group.values(),
        label="Criteria",
        description="The function to measure the quality of a split",
        view=types.DropdownView(),
        default="gini",
    )


def _configure_gradient_boosting_classifier(inputs):
    inputs.view(
        "gb_header",
        types.Header(
            label="Gradient Boosting Classifier",
            description="Configure the Gradient Boosting Classifier",
            divider=True,
        ),
    )

    inputs.int("n_estimators", label="Number of estimators", default=100)
    inputs.int("max_depth", label="Max depth", default=3)
    inputs.float("learning_rate", label="Learning rate", default=0.1)
    inputs.float("subsample", label="Subsample", default=1.0)
    inputs.float("min_samples_split", label="Min samples split", default=2)
    inputs.float("min_samples_leaf", label="Min samples leaf", default=1)
    inputs.float("max_leaf_nodes", label="Max leaf nodes", default=None)


def _configure_bagging_classifier(inputs):
    inputs.view(
        "bg_header",
        types.Header(
            label="Bagging Classifier",
            description="Configure the Bagging Classifier",
            divider=True,
        ),
    )

    estimator_choices = ["decision_tree", "knn", "svm"]
    estimator_group = types.RadioGroup()
    for choice in estimator_choices:
        estimator_group.add_choice(choice, label=choice)

    inputs.enum(
        "estimator",
        estimator_group.values(),
        label="Base estimator",
        description="The base estimator to fit on random subsets of the dataset",
        view=types.DropdownView(),
        default="decision_tree",
    )

    inputs.int("n_estimators", label="Number of estimators", default=10)
    inputs.float("max_samples", label="Max samples", default=1.0)
    inputs.float("max_features", label="Max features", default=1.0)
    inputs.bool("bootstrap", label="Bootstrap", default=True)
    inputs.bool(
        "bootstrap_features", label="Bootstrap features", default=False
    )


def _configure_adaboost_classifier(inputs):
    inputs.view(
        "ab_header",
        types.Header(
            label="AdaBoost Classifier",
            description="Configure the AdaBoost Classifier",
            divider=True,
        ),
    )

    estimator_choices = ["decision_tree", "knn", "svm"]
    estimator_group = types.RadioGroup()
    for choice in estimator_choices:
        estimator_group.add_choice(choice, label=choice)

    inputs.enum(
        "estimator",
        estimator_group.values(),
        label="Base estimator",
        description="The base estimator from which the boosted ensemble is built.",
        view=types.DropdownView(),
        default="decision_tree",
    )

    inputs.int("n_estimators", label="Number of estimators", default=50)
    inputs.float("learning_rate", label="Learning rate", default=1.0)


class CreateLearner(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="create_learner",
            label="Active Learning: create learner",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Active Learning",
            description="Label samples with Active Learning",
        )
        if TEAMS_DEPLOYMENT:
            return types.Property(inputs, view=form_view)

        cand_feature_fields = _get_candidate_feature_fields(ctx.dataset)
        if not _ensure_feature_fields(cand_feature_fields, inputs):
            return types.Property(inputs, view=form_view)

        _initial_labels_selection(inputs)

        if ctx.params.get("init_labels", "labels") == "tags":
            if not _ensure_tags(ctx.dataset, inputs):
                return types.Property(inputs, view=form_view)
        else:
            classif_fields = _get_classification_fields(ctx.dataset)
            if len(classif_fields) == 0:
                inputs.view(
                    "warning",
                    types.Warning(
                        label="No Classification Fields",
                        description=(
                            "You must have classification fields on your dataset."
                        ),
                    ),
                )

                return types.Property(inputs, view=form_view)
            elif len(classif_fields) == 1:
                cf = classif_fields[0]
                ctx.params["init_label_field"] = cf
                inputs.str(
                    f"init_label_field_message_{cf}",
                    description=f"Field `{cf}` will be used as initial labels",
                    view=types.MarkdownView(read_only=True),
                )
            else:
                field_dropdown = types.AutocompleteView(label="Label Field")
                for cf in classif_fields:
                    field_dropdown.add_choice(cf, label=cf)

                inputs.enum(
                    "init_label_field",
                    field_dropdown.values(),
                    view=field_dropdown,
                )

        inputs.view(
            "features_header",
            types.Header(
                label="Input features",
                description="Select the features to use as input to the model",
                divider=True,
            ),
        )

        if len(cand_feature_fields) == 1:
            ctx.params["feature_fields"] = cand_feature_fields
        else:
            for ff in cand_feature_fields:
                inputs.bool(ff, label=ff, default=False)

            feature_fields = ctx.params.get("feature_fields", [])
            if len(feature_fields) == 0:
                cand_feature_fields = _get_candidate_feature_fields(
                    ctx.dataset
                )
                feature_fields = [
                    cf
                    for cf in cand_feature_fields
                    if ctx.params.get(cf, False)
                ]
                ctx.params["feature_fields"] = feature_fields

            if len(feature_fields) == 0:
                inputs.view(
                    "warning",
                    types.Warning(
                        label="No input features",
                        description=(
                            "You must select at least one input feature. If you"
                            "select more than one, they will be concatenated."
                        ),
                    ),
                )

        inputs.view(
            "learner_header",
            types.Header(
                label="Learner Details",
                description="Configure the learner",
                divider=True,
            ),
        )

        inputs.str(
            "labels_field",
            label="Prediction field",
            description="The field to store the predictions in",
            required=True,
        )
        inputs.int("batch_size", label="Batch size", default=5)

        learner_labels = [
            "Random Forest",
            "Gradient Boosting",
            "Bagging",
            "AdaBoost",
        ]
        learner_group = types.RadioGroup()
        for llabel in learner_labels:
            learner_group.add_choice(llabel, label=llabel)

        inputs.enum(
            "classifier_type",
            learner_group.values(),
            label="Learner",
            description="The learner to use",
            view=types.DropdownView(),
            default="Random Forest",
        )

        if (
            ctx.params.get("classifier_type", "Random Forest")
            == "Random Forest"
        ):
            _configure_random_forest_classifier(inputs)
        elif (
            ctx.params.get("classifier_type", "Random Forest")
            == "Gradient Boosting"
        ):
            _configure_gradient_boosting_classifier(inputs)
        elif ctx.params.get("classifier_type", "Random Forest") == "Bagging":
            _configure_bagging_classifier(inputs)
        elif ctx.params.get("classifier_type", "Random Forest") == "AdaBoost":
            _configure_adaboost_classifier(inputs)

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        initialize_learner(ctx)
        ctx.trigger("reload_dataset")


class QueryLearner(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="query_learner",
            label="Active Learning: query learner",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Active Learning",
            description="Label samples with Active Learning",
        )
        if TEAMS_DEPLOYMENT:
            return types.Property(inputs, view=form_view)

        inputs.int(
            "batch_size",
            label="Batch size",
            description="Override the batch size for this query",
            default=None,
        )
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        batch_size = ctx.params.get("batch_size", None)
        sample_ids = query_learner(ctx.dataset, batch_size=batch_size)
        view = ctx.dataset.select(sample_ids, ordered=True)
        ctx.trigger(
            "set_view",
            params=dict(view=serialize_view(view)),
        )
        return


class TeachLearner(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="update_learner_predictions",
            label="Active Learning: teach learner and update predictions",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Active Learning",
            description="Label samples with Active Learning",
        )
        if TEAMS_DEPLOYMENT:
            return types.Property(inputs, view=form_view)

        cand_feature_fields = _get_candidate_feature_fields(ctx.dataset)
        if not _ensure_feature_fields(cand_feature_fields, inputs):
            return types.Property(inputs, view=form_view)

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        teach_learner(ctx.dataset)
        predict(ctx.dataset)
        ctx.trigger("reload_dataset")
        return


def register(plugin):
    plugin.register(CreateLearner)
    plugin.register(QueryLearner)
    plugin.register(TeachLearner)
