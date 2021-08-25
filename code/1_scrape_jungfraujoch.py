import os
import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import ssl

from tqdm import tqdm
from lightly.utils import save_custom_metadata

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    size = ['s', 'm', ''][1]  # small, medium or large size files
    output_dir = os.environ.get("WEATHER_DIR_RAW_LOCAL")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file_metadata = os.path.join(output_dir, "weather_jungfraujoch_metadata.json")


    metadata_list = []

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    delta = datetime.timedelta(days=1)
    difference_total = end_date-start_date
    images_to_take = int(difference_total/delta)
    for iter in tqdm(range(images_to_take+1)):
        current_date = start_date + delta * iter
        year = current_date.year
        month = current_date.month
        day = current_date.day

        filename = f"Jungfraujoch_{size}{year:04d}{month:02d}{day:02d}1300.jpg"
        url = f"https://webcam.switch.ch/jungfraujoch/pano/{year:04d}/{month:02d}{day:02d}/{filename}"
        filepath = os.path.join(output_dir, filename)
        try:
            urlretrieve(url, filepath)
            metadata_list.append((filename, {'days_from_start': int(iter)}))
        except HTTPError:
            print(f"Skipping image at {url}")

    save_custom_metadata(output_file_metadata, metadata_list)
    print(f"Path to metadata file: {output_file_metadata}")

    print(f"Use the following command in the next step:")

    print("Command with training on GPU yourself:")
    lightly_command = f"lightly-magic input_dir={output_dir} new_dataset_name=Weather_Jungfraujoch " \
                      f"custom_metadata={output_file_metadata} trainer.max_epochs=400 loader.batch_size=176 token=MY_TOKEN"
    print(lightly_command)

    print("RECOMMENDED: Command using embeddings from pretrained model.")
    lightly_command =f"lightly-magic input_dir={output_dir} new_dataset_name=Weather_Jungfraujoch " \
                     f"custom_metadata={output_file_metadata} trainer.max_epochs=0 token=MY_TOKEN"
    print(lightly_command)


