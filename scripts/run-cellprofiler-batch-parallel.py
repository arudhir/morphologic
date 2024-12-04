import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

# base directories
base_dir = "/mnt/data/egfr-images3/EGFR_KO_and_controls"
output_dir = "/mnt/data/cellprofiler-output"
pipeline_file = "/home/ubuntu/morphologic/pipelines/rxrx3.cppipe"

# ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# function to run cellprofiler for a given directory
def run_cellprofiler(input_dir):cd 
    # extract the plate number from the directory name
    plate = next((part for part in input_dir.split("/") if "Plate" in part), "NoPlate")

    # create output directory for this plate
    plate_output_dir = os.path.join(output_dir, plate)
    os.makedirs(plate_output_dir, exist_ok=True)

    # construct the CellProfiler command
    command = [
        "cellprofiler", "-c", "-r",
        "--pipeline", pipeline_file,
        "-i", input_dir,
        "-o", plate_output_dir,
    ]

    # execute the command and capture output
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Completed: {input_dir}\n{result.stdout}")
        print(f"Output can be found in {plate_output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error in {input_dir}: {e.stderr}")

# find all directories containing .png files and not already processed
def find_input_dirs(base_dir):
    input_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # check if directory contains .png files
        contains_png = any(file.endswith(".png") for file in files)
        
        # check if directory has "overlay_images/" subfolder
        already_processed = "overlay_images" in dirs

        # add to list if it contains .png and is not processed
        if contains_png and not already_processed:
            input_dirs.append(root)
    return input_dirs

# main function to parallelize
def main():
    input_dirs = find_input_dirs(base_dir)
    print(f"Found {len(input_dirs)} directories to process.")

    # use ProcessPoolExecutor to run in parallel
    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(run_cellprofiler, input_dirs)

if __name__ == "__main__":
    main()

