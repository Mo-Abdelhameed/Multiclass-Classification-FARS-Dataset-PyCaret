import pandas as pd
from config import paths
from utils import read_csv_in_directory, save_dataframe_as_csv
from logger import get_logger
from Classifier import Classifier
from schema.data_schema import load_saved_schema, MulticlassClassificationSchema
from preprocessing import preprocess

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
        predictions_df: pd.DataFrame,
        schema: MulticlassClassificationSchema,
        return_probs: bool = False,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Args:
        predictions_df (predictions_df): Predicted probabilities from predictor model.
        schema (MulticlassClassificationSchema): schema of the data used in training.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    ids = predictions_df[schema.id]
    predictions_df = predictions_df.iloc[:, -len(schema.target_classes) - 1:]
    predictions_df.insert(0, schema.id, ids)
    columns = []
    for c in predictions_df.columns:
        if c.startswith('prediction_score_'):
            columns.append(c.replace('prediction_score_', "", 1))
        else:
            columns.append(c)
    predictions_df.columns = columns
    if return_probs:
        predictions_df = predictions_df.drop(columns=['prediction_label'])
        return predictions_df
    return predictions_df[[schema.id, 'prediction_label']]


def run_batch_predictions() -> None:
    """
        Run batch predictions on test data, save the predicted probabilities to a CSV file.

        This function reads test data from the specified directory,
        loads the preprocessing pipeline and pre-trained predictor model,
        transforms the test data using the pipeline,
        makes predictions using the trained predictor model,
        adds ids into the predictions dataframe,
        and saves the predictions as a CSV file.
        """
    x_test = preprocess(paths.TEST_DIR, training=False)
    data_schema = load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH)
    model = Classifier.load(paths.PREDICTOR_DIR_PATH)
    logger.info("Making predictions...")
    predictions_df = Classifier.predict_with_model(model, x_test, raw_score=True)
    predictions_df = create_predictions_dataframe(
        predictions_df,
        data_schema,
        return_probs=False,
    )

    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=paths.PREDICTIONS_FILE_PATH
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
