from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasourcePurpose

# Create the Lightly client to connect to the Lightly Platform.
client = ApiWorkflowClient(token="YOUR_LIGHTLY_TOKEN")

# Set the dataset to the one we created.
client.set_dataset_id_by_name(dataset_name="weather-classification")

# Configure the Input datasource.
client.set_s3_delegated_access_config(
    resource_path="s3://<your_bucket>/data/",
    region="your_bucket_region",
    role_arn="your_role_arn",
    external_id="your_external_id",
    purpose=DatasourcePurpose.INPUT,
)
# Configure the Lightly datasource.
client.set_s3_delegated_access_config(
    resource_path="s3://<your_bucket>/lightly/",
    region="your_bucket_region",
    role_arn="your_role_arn",
    external_id="your_external_id",
    purpose=DatasourcePurpose.LIGHTLY,
)

# Create a Lightly Worker run to select another 30 samples using active learning.
scheduled_run_id = client.schedule_compute_worker_run(
    worker_config={
        "datasource": {
            "process_all": True,
        },
        "enable_training": False,
    },
    selection_config={
        "n_samples": 30,
        "strategies": [
            {
                "input": {
                    "type": "SCORES",
                    "task": "weather-classification",
                    "score": "uncertainty_entropy",
                },
                "strategy": {"type": "WEIGHTS"},
            }
        ],
    },
)

for run_info in client.compute_worker_run_info_generator(
    scheduled_run_id=scheduled_run_id
):
    print(
        f"Lightly Worker run is now in state='{run_info.state}' with message='{run_info.message}'"
    )

if run_info.ended_successfully():
    print("SUCCESS")
else:
    print("FAILURE")
