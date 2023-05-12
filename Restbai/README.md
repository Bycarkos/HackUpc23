# HackUPC 2023 Restb.ai dataset

## ZIP Content

- `hackupc2023_restbai__dataset.json`: 1.3 GB JSON file with the whole dataset (658,676 properties and 6,350,927 images).
- `hackupc2023_restbai__dataset_sample.json`: Random set of 100 sample properties for visually analyzing the dataset.
- `about_restbai_solutions.pdf`: 6-slide PDF adding more context about the different information about the Restb.ai solutions.
  - Even more details can be found online in our official API documentation: https://docs.restb.ai/vision

## Notes

We encourage hackers to analyze first the Sample version, in order to understand the structure of the dataset. Once that, you can go over the actual dataset.

And, obviously, any questions (which are expected), feel free to ask through Slack or come by our booth.

## JSON structure

Once the JSON file is loaded, you can find a dictionary where the `key` is the property ID and the `value` is the information of that property. Each property ID is always an integer value.

Each property value can be described as is:

- `summary`: Title describing the property (written in Spanish).
- `city`: Municipality where the property is located.
- `neighborhood`: Neighborhood where the property is located.
- `region`: Region / Province where the property is located.
- `price`: Sold price (in Euros) of the property once was listed.
- `square_meters`: Living area of the property.
- `bedrooms`: Number of bedrooms of the property.
- `bathrooms`: Number of bathrooms on the property.
- `images`: Array of Internet-accessible URLs pointing to the property images.
  - Format is always like `https://restb-hackathon.s3.amazonaws.com/real_estate_dataset/images/{PROPERTY_ID}__{IMAGE_IDX}.jpg` where `PROPERTY_ID` is the same as the `key` of the dictionary and the `IMAGE_IDX` is going from `0` to `n - 1`.
- `num_images`: Number of images available of the property.
- `image_data`: Dictionary with the image insights that Restb.ai solutions could extract:
  - `image_data.r1r6`: Dictionary with the 4 different values of the R1R6 score (see PDF attached into the ZIP for more context).
    - `image_data.r1r6.property`: R1R6 score of the whole property.
    - `image_data.r1r6.kitchen`: R1R6 score of the kitchen sub-group of the property.
    - `image_data.r1r6.bathroom`: R1R6 score of the bathroom sub-group of the property.
    - `image_data.r1r6.interior`: R1R6 score of the interior sub-group of the property.
  - `image_data.style`: Dictionary with the Exterior Style of the property.
    - `image_data.style.label`: Label of the most confident prediction of the style of the property.
    - `image_data.style.confidence`: Float value (0 - 1) related with the confidence of the style of the property.
  - `image_data.features_by_room_type`: Dictionary with the Features found in the property split by Room types.
    - Each `key` of this dictionary is the Room Type label (i. e. `kitchen`).
    - Each value is a dictionary with the information found inside:
      - `image_data.features_by_room_type.RT.unique_features`: List of dictionaries with all the unique features from a list of +100 available options.
        - `image_data.features_by_room_type.RT.unique_features.label`: Label of the detection found in the `RT` room type (i. e. `refrigerator`).
        - `image_data.features_by_room_type.RT.unique_features.details`: Array of detailed labels related to the main detection (i. e. `stainless_steel`).
      - `image_data.features_by_room_type.RT.best_photo_idx`: Integer pointing to the index of the `images` array referring to the best photo of that room type of the property.
- `property_type`: Property type of the property.
  - Within the following 4 options: `single_family`, `condo_apartment`, `multi_family`, and `townhome`.