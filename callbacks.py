import importlib

from transformers.utils import logging
from transformers.integrations import TrainerCallback


logger = logging.get_logger(__name__)


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


class MLflowCustomCallback(TrainerCallback):
    def __init__(self, run, experiment, log_artifacts):
        assert is_mlflow_available(), "MLflowCallback requires mlflow to be installed. Run `pip install mlflow`."
        import mlflow

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._log_artifacts = False
        self._ml_flow = mlflow
        self._log_artifacts = log_artifacts
        self._run_name = run
        self._experiment = experiment

    def setup(self, args, state, model):
        if state.is_world_process_zero:
            self._ml_flow.set_experiment(self._experiment)
            self._ml_flow.start_run(run_name=self._run_name)
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        self._initialized = True


    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self._ml_flow.log_metric(k, v, step=state.global_step)
                else:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a metric. '
                        f"MLflow's log_metric() only accepts float and "
                        f"int types so we dropped this attribute."
                    )

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if self._ml_flow.active_run is not None:
            self._ml_flow.end_run()
