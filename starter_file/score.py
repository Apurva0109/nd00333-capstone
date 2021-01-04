# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Customer_Age": pd.Series([0.0], dtype="float64"), "Gender": pd.Series([0.0], dtype="float64"), "Dependent_count": pd.Series([0.0], dtype="float64"), "Education_Level": pd.Series([0.0], dtype="float64"), "Income_Category": pd.Series([0.0], dtype="float64"), "Months_on_book": pd.Series([0.0], dtype="float64"), "Total_Relationship_Count": pd.Series([0.0], dtype="float64"), "Months_Inactive_12_mon": pd.Series([0.0], dtype="float64"), "Contacts_Count_12_mon": pd.Series([0.0], dtype="float64"), "Credit_Limit": pd.Series([0.0], dtype="float64"), "Total_Revolving_Bal": pd.Series([0.0], dtype="float64"), "Avg_Open_To_Buy": pd.Series([0.0], dtype="float64"), "Total_Amt_Chng_Q4_Q1": pd.Series([0.0], dtype="float64"), "Total_Trans_Amt": pd.Series([0.0], dtype="float64"), "Total_Trans_Ct": pd.Series([0.0], dtype="float64"), "Total_Ct_Chng_Q4_Q1": pd.Series([0.0], dtype="float64"), "Avg_Utilization_Ratio": pd.Series([0.0], dtype="float64"), "Marital_Status_Divorced": pd.Series([0.0], dtype="float64"), "Marital_Status_Married": pd.Series([0.0], dtype="float64"), "Marital_Status_Single": pd.Series([0.0], dtype="float64"), "Card_Category_Blue": pd.Series([0.0], dtype="float64"), "Card_Category_Gold": pd.Series([0.0], dtype="float64"), "Card_Category_Platinum": pd.Series([0.0], dtype="float64"), "Card_Category_Silver": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
