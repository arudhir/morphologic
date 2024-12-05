#!/usr/bin/env python
import argparse
import pandas as pd
import json
import os

def csv_to_geojson(csv_file, output_dir):
    # load the csv
    df = pd.read_csv(csv_file)

    # check for required coordinate columns
    if 'Location_Center_X' not in df.columns or 'Location_Center_Y' not in df.columns:
        print(f"skipping {csv_file}: missing coordinate columns.")
        return

    # generate geojson features
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row['Location_Center_X'], row['Location_Center_Y']]},
            "properties": row.drop(['Location_Center_X', 'Location_Center_Y']).to_dict()
        }
        for _, row in df.iterrows()
    ]

    # assemble geojson structure
    geojson = {"type": "FeatureCollection", "features": features}

    # output file name
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(csv_file))[0]}.geojson")
    with open(output_file, "w") as f:
        json.dump(geojson, f)
    print(f"geojson saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="convert cellprofiler csv files to geojson.")
    parser.add_argument("--csvs", nargs="+", required=True, help="list of csv files to process.")
    parser.add_argument("--output_dir", default=".", help="output directory for geojson files (default: current directory).")
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # process each csv
    for csv_file in args.csvs:
        csv_to_geojson(csv_file, args.output_dir)

if __name__ == "__main__":
    main()

