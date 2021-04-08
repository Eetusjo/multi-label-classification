import importlib
import mlflow
import mlflow.pyfunc as pyfunc
import models
import numpy as np
import os

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from models import MLFlowBertClassificationModel
from transformers.utils import logging
from transformers.integrations import TrainerCallback


logger = logging.get_logger(__name__)


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


class MLflowCustomCallback(TrainerCallback):
    def __init__(self,
                 run,
                 experiment,
                 register_best,
                 tracking_uri="http://localhost:5000"):
        assert is_mlflow_available(), "MLflowCallback requires mlflow to be installed. Run `pip install mlflow`."

        mlflow.set_tracking_uri(tracking_uri)

        self.MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self.MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self.initialized = False
        self.ml_flow = mlflow
        self.register_best = register_best
        self.run_name = run
        self.experiment = experiment

    def setup(self, args, state, model):
        if state.is_world_process_zero:
            self.ml_flow.set_experiment(self.experiment)
            self.ml_flow.start_run(run_name=self.run_name)
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self.MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self.MAX_PARAMS_TAGS_PER_BATCH):
                self.ml_flow.log_params(dict(combined_dict_items[i : i + self.MAX_PARAMS_TAGS_PER_BATCH]))
        self.initialized = True


    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self.initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self.initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.ml_flow.log_metric(k, v, step=state.global_step)
                else:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a metric. '
                        f"MLflow's log_metric() only accepts float and "
                        f"int types so we dropped this attribute."
                    )

    def on_train_end(self, args, state, control, **kwargs):
        input_schema = Schema([ColSpec(name="text", type="string")])
        output_schema = Schema([TensorSpec(np.dtype(np.float), (-1, -1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        pyfunc.log_model(
            # artifact path is _relative_ to run root in mlflow
            artifact_path="bert_classifier_model",
            # Dir with the module files for dependencies
            code_path=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.py"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")
            ],
            python_model=MLFlowBertClassificationModel(),
            artifacts={
                "model": state.best_model_checkpoint,
            },
            conda_env={
                'name': 'classifier-env',
                'channels': ['defaults', 'pytorch', 'pypi'],
                'dependencies': [
                    'python=3.8.8',
                    'pip',
                    'pytorch=1.8.0',
                    {'pip': [
                        'transformers==4.4.2',
                        'mlflow==1.15.0',
                        'numpy==1.20.1'
                     ]}
                ]
            },
            signature=signature,
            await_registration_for=0
        )

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if self.ml_flow.active_run is not None:
            self.ml_flow.end_run()
