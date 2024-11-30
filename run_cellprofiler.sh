#!/bin/bash
input_dir=$1
ls "$input_dir" | parallel --jobs 16 cellprofiler -c -r --pipeline /home/ubuntu/morphologic/pipelines/rxrx3.cppipe -i ${input_dir}/{} -o /mnt/data/cellprofiler-output/{}

