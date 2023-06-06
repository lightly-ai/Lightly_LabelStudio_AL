import pathlib

import requests
from lightly.api import ApiWorkflowClient

# Create the Lightly client to connect to the Lightly Platform.
client = ApiWorkflowClient(token="YOUR_LIGHTLY_TOKEN")

# Set the dataset to the one we created.
client.set_dataset_id_by_name(dataset_name="weather-classification")
latest_tag = client.get_all_tags()[0]

# filename_url_mappings is a list of entries with their filenames and read URLs.
# For example, [{"fileName": "image1.png", "readUrl": "https://..."}]
filename_url_mappings = client.export_filenames_and_read_urls_by_tag_id(latest_tag.id)

output_path = pathlib.Path("samples_for_labelling")
output_path.mkdir(exist_ok=True)

for entry in filename_url_mappings:
    read_url = entry["readUrl"]
    filename = entry["fileName"]
    print(f"Downloading {filename}")
    response = requests.get(read_url, stream=True)
    with open(output_path / filename, "wb") as file:
        for data in response.iter_content():
            file.write(data)
