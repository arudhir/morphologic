#!/usr/bin/env python
import argparse
import pandas as pd
import json
import os

def csv_to_geojson_with_bbox(csv_file, output_dir):
    # load the csv
    df = pd.read_csv(csv_file)

    # check for required bounding box columns
    required_columns = [
        'AreaShape_BoundingBoxMaximum_X', 
        'AreaShape_BoundingBoxMaximum_Y',
        'AreaShape_BoundingBoxMinimum_X', 
        'AreaShape_BoundingBoxMinimum_Y'
    ]
    if not all(col in df.columns for col in required_columns):
        print(f"skipping {csv_file}: missing bounding box columns.")
        return

    # generate geojson features with bounding boxes
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [row['AreaShape_BoundingBoxMinimum_X'], row['AreaShape_BoundingBoxMinimum_Y']],
                    [row['AreaShape_BoundingBoxMaximum_X'], row['AreaShape_BoundingBoxMinimum_Y']],
                    [row['AreaShape_BoundingBoxMaximum_X'], row['AreaShape_BoundingBoxMaximum_Y']],
                    [row['AreaShape_BoundingBoxMinimum_X'], row['AreaShape_BoundingBoxMaximum_Y']],
                    [row['AreaShape_BoundingBoxMinimum_X'], row['AreaShape_BoundingBoxMinimum_Y']]  # close the loop
                ]]
            },
            "properties": row.drop(required_columns).to_dict()  # include other metadata
        }
        features.append(feature)

    # assemble geojson structure
    geojson = {"type": "FeatureCollection", "features": features}

    # output file name
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(csv_file))[0]}.geojson")
    with open(output_file, "w") as f:
        json.dump(geojson, f)
    print(f"geojson saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="convert cellprofiler csv files to geojson with bounding boxes.")
    parser.add_argument("--csvs", nargs="+", required=True, help="list of csv files to process.")
    parser.add_argument("--output_dir", default=".", help="output directory for geojson files (default: current directory).")
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # process each csv
    for csv_file in args.csvs:
        csv_to_geojson_with_bbox(csv_file, args.output_dir)

if __name__ == "__main__":
    main()
