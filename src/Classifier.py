import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError
from schema.data_schema import MulticlassClassificationSchema
from sklearn.metrics import f1_score

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from pycaret.classification import compare_models, setup, finalize_model, predict_model, add_metric, remove_metric

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = 'predictor.joblib'


def macro_f1_score(y, y_pred, **kwargs):
    return f1_score(y_pred=y_pred, y_true=y, average='macro')


class Classifier:
    """A wrapper class for the multiclass classifier.

        This class provides a consistent interface that can be used with other
        classifier models.
    """

    model_name = 'pycaret_multiclass_classifier'

    def __init__(self, train_input: pd.DataFrame, schema: MulticlassClassificationSchema):
        """Construct a new Binary Classifier."""
        self._is_trained = False
        self.schema = schema
        self.setup(train_input, schema)
        self.model = self.compare_models()

    def compare_models(self):
        """Build a new binary classifier."""
        return compare_models(include=[
                                    # 'xgboost',
                                    'et',
                                    # 'rf',
        ])

    def setup(self, train_input: pd.DataFrame, schema: MulticlassClassificationSchema):
        """Set up the experiment of comparing different models.

        Args:
            train_input: The data  of training including the target column.
            schema: schema of the provided data.
        """
        setup(train_input, target=schema.target, remove_outliers=True, normalize=True, ignore_features=[schema.id], train_size=0.98)
        add_metric('Macro F1', 'Macro F1', score_func=macro_f1_score, greater_is_better=True)
        remove_metric('kappa')
        remove_metric('mcc')
        remove_metric('f1')
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        pipeline = finalize_model(self.model)
        joblib.dump(pipeline, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    @classmethod
    def predict_with_model(cls, classifier: "Classifier", data: pd.DataFrame, raw_score=True
                           ) -> pd.DataFrame:
        """
        Predict class probabilities for the given data.

        Args:
            classifier (Classifier): The classifier model.
            data (pd.DataFrame): The input data.
            raw_score (bool): Whether to return class probabilities or labels.
                Defaults to True.

        Returns:
            np.ndarray: The predicted classes or class probabilities.
        """
        return predict_model(classifier, data, raw_score=raw_score)

    @classmethod
    def save_predictor_model(cls, model: "Classifier", predictor_dir_path: str) -> None:

        """
        Save the classifier model to disk.

        Args:
            model (Classifier): The classifier model to save.
            predictor_dir_path (str): Dir path to which to save the model.
        """
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)

    @classmethod
    def load_predictor_model(cls, predictor_dir_path: str) -> "Classifier":
        """
        Load the classifier model from disk.

        Args:
            predictor_dir_path (str): Dir path where model is saved.

        Returns:
            Classifier: A new instance of the loaded classifier model.
        """
        return Classifier.load(predictor_dir_path)


