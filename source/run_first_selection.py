from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType, DatasourcePurpose

# Create the Lightly client to connect to the Lightly Platform.
client = ApiWorkflowClient(token="YOUR_LIGHTLY_TOKEN")

# Create a new dataset on the Lightly Platform.
client.create_dataset(
    dataset_name="weather-classification", dataset_type=DatasetType.IMAGES
)
dataset_id = client.dataset_id

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

# Create a Lightly Worker run to select the first batch of 30 samples
# based on image embeddings.
scheduled_run_id = client.schedule_compute_worker_run(
    selection_config={
        "n_samples": 30,
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS"},
                "strategy": {"type": "DIVERSITY"},
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
