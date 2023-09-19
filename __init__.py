"""Active Learning plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
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
        from active_learning import initialize_learner, query_learner, teach_learner, predict


def get_vector_fields(dataset):
    """Get all vector fields in a dataset."""
    vector_fields = []
    fields = dataset.get_field_schema(flat=True)
    for field, ftype in fields.items():
        full_type = str(ftype)
        if "VectorField" in full_type:
            vector_fields.append(field)
    return vector_fields


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _ensure_embeddings(vector_fields, inputs):
    if len(vector_fields) == 0:
        inputs.view(
            "warning", 
            types.Warning(
                label="No Embeddings", 
                description=(
                    "To use active learning, you must embeddings on your dataset."
                    " You can create one by running `dataset.compute_embeddings()`."
                )
            )
        )
        return False
    return True
    

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
            label="Active Learning", description="Label samples with Active Learning"
        )
        if TEAMS_DEPLOYMENT:
            return types.Property(inputs, view=form_view)
        
        vector_fields = get_vector_fields(ctx.dataset)
        if not _ensure_embeddings(vector_fields, inputs):
            return types.Property(inputs, view=form_view)

        field_dropdown = types.AutocompleteView(label="Embedding Field")
        for vf in vector_fields:
            field_dropdown.add_choice(vf, label=vf)

        inputs.enum(
            "embeddings_field",
            field_dropdown.values(),
            view=field_dropdown,
        )

        inputs.str("labels_field", label="Labels field", required=True)
        inputs.int("batch_size", label="Batch size", default=5)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        embeddings_field = ctx.params["embeddings_field"]
        labels_field = ctx.params["labels_field"]
        batch_size = ctx.params["batch_size"]
        initialize_learner(
            ctx.dataset,
            embeddings_field,
            labels_field,
            batch_size=batch_size,
        )
        return
    

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
            label="Active Learning", description="Label samples with Active Learning"
        )
        if TEAMS_DEPLOYMENT:
            return types.Property(inputs, view=form_view)
        
        vector_fields = get_vector_fields(ctx.dataset)
        if not _ensure_embeddings(vector_fields, inputs):
            return types.Property(inputs, view=form_view)

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        sample_ids = query_learner(ctx.dataset)
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
            label="Active Learning", description="Label samples with Active Learning"
        )
        if TEAMS_DEPLOYMENT:
            return types.Property(inputs, view=form_view)
        
        vector_fields = get_vector_fields(ctx.dataset)
        if not _ensure_embeddings(vector_fields, inputs):
            return types.Property(inputs, view=form_view)

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        teach_learner(ctx.dataset)
        predict(ctx.dataset)
        ctx.trigger('reload_dataset')
        return
    

def register(plugin):
    plugin.register(CreateLearner)
    plugin.register(QueryLearner)
    plugin.register(TeachLearner)
