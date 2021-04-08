import argparse
import logging
import mlflow
import mlflow.pyfunc
import numpy as np
import sys

from flask import Flask, request
from flask_apscheduler import APScheduler
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

scheduler = APScheduler()
scheduler.api_enabled = True
scheduler.app = app
scheduler.init_app(app)
scheduler.start()

logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)

app = Flask(__name__)


@app.route('/', methods=["POST"])
def predict():
    return app.model.predict()


class Model(object):
    def __init__(self, client, model_name, stage, version):
        self.client = client
        self.model = None
        self.model_version = None
        self.update_model(model_name, stage, version)

    def predict(self):
        return self.model.predict(np.array([[1, 2, 3]]))

    def update_model(self, model_name, stage=None, version=None):
        if version:
            scheduler.pause()
            self.model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{version}"
            )
            self.model_version = version
            app.logger.info("Succesfully loaded model:\n" + str(self.model))
            scheduler.resume()
        else:
            latest = None
            models = self.client.search_registered_models(f"name='{model_name}'")[0]
            for model in models.latest_versions:
                if model.current_stage == stage:
                    latest = model
                    latest_version = int(latest.version)
            if latest is None:
                app.logger.warning(f"Found no models in stage '{stage}'.")
                return

            if (self.model_version is None) or (latest_version > self.model_version):
                app.logger.info(f"Found newer model in stage '{stage}'")
                self.update_model(model_name, version=latest_version)
            elif latest_version < self.model_version:
                app.logger.info(f"Seems like current model has been rolled "
                                f"back. Downgrading model.")
                self.update_model(model_name, version=latest_version)


def main(argv):
    parser = argparse.ArgumentParser('serve')
    parser.add_argument("--model", required=True,
                        help="model name in model registry")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", default=None,
                        help="use model with this stage")
    group.add_argument("--version", default=None,
                        help="use this model version")

    parser.add_argument("--auto_update_model", action="store_true",
                        help="If using a model from a specific stage, "
                             "auto-update whenever a new model is registered "
                             "to that stage")

    args = parser.parse_args()

    if args.auto_update_model:
        assert args.stage, "'auto_update_model' can only be used with 'stage'"

    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    app.model = Model(client, args.model, stage=args.stage, version=args.version)

    if args.auto_update_model:
        @scheduler.task('interval', id='do_job_1', seconds=10, misfire_grace_time=900)
        def maybe_update_model():
            global app
            app.model.update_model(args.model, args.stage, None)

    app.run(host='0.0.0.0', port=5050)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
