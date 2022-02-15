

import mlflow
experiment_id = "some_experiment_id"

with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("dropout", 0.25)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("n_layers", 5)

    mlflow.log_metric("precision", 0.76)
    mlflow.log_metric("recall", 0.92)
    mlflow.log_metric("f1", 0.83)
    mlflow.log_metric("coverage", 0.76)

#2-LOG DATA IN NESTED RUNS
with mlflow.start_run(experiment_id=experiment_id) as run:
    with mlflow.start_run(experiment_id=experiment_id, nested=True) as nested_run:
        mlflow.log_metric("f1", 0.29)
        mlflow.log_param("accuracy", 0.19)


#3- Get a run based on id and access its data
from mlflow.tracking import MlflowClient
client = MlflowClient()
run_id = "5aa1f947312a44c68c844bc4034497d7"

run = client.get_run(run_id)
print(run)


#4 - Filter runs based on search queries
import random
def generate_random_params():
    lr = random.random()
    dropout = random.random()
    optimizer = random.choice(["sgd", "adam", "adamw", "rmsprop"])
    n_layers = random.randint(1, 20)

    return {
        "lr": lr,
        "dropout": dropout,
        "optimizer": optimizer,
        "n_layers": n_layers,
    }

def generate_random_metrics():
    precision = random.random()
    recall = random.random()
    f1 = (2 * precision * recall) / (precision + recall)
    coverage = random.random()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
    }


for _ in range(50):
    params = generate_random_params()
    metrics = generate_random_metrics()

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


##5-UPLOAD ARTIFACTS
with mlflow.start_run(experiment_id=experiment_id) as run:
    params = generate_random_metrics()
    metrics = generate_random_metrics()

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.log_artifact("./stats_comparison.csv")


##6-DOWNLOAD ARTIFACTS
run_id = "78a0e1927ac5473eb79125ed7d6ebee6"
client.download_artifacts(run_id=run_id, path=".", dst_path="./downloads/")


#7-UPDATE AN EXISTING RUN
run_id = "f0a285ab628245a79f417ab0706b9a99"

with mlflow.start_run(run_id=run_id):
    random_metrics = generate_random_metrics()
    mlflow.log_metrics(random_metrics)



from xlwings import Workbook, Range, Sheet
import pandas as pd

# Split Excel data in one worksheet into multiple worksheets based on column name.

# Copy this file into the same location as the Excel workbook with the worksheet you wish to split.
# Download the zip of the xlwings Github repo here: https://github.com/ZoomerAnalytics/xlwings and copy the
# xlwings.bas file from the xlwings folder. Import the xlwings.bas file into your Excel workbook by entering ALT+F11
# and then going to File, Import File and clicking on the file.
# Import the Split_Excel_Worksheet.bas file and run by going to the Developer tab on the Excel Ribbon, click Macros,
# and select Split_Excel_Workbooks

# Make sure the data you want split starts in cell A1 and has no empty column headers.
# The worksheet to split should be the first worksheet in the workbook.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main Script
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def split_worksheets():
    wb = Workbook.caller()
    sheet = Range('temp', 'AA1').value
    column = Range('temp', 'AA2').value

    data = pd.DataFrame(pd.read_excel(sheet, 0, index_col=None, na_values=[0]))
    data.sort(column, axis = 0, inplace = True)

    split = data.groupby(column)
    for i in split.groups:
        Sheet.add()
        Range('A1', index=False).value = split.get_group(i)

