#!/bin/bash
url="$1"
echo "Processing URL: $url"
plateID=$(echo "$url" | awk -F"/" '{print $NF}' | awk -F"?" '{print $1}')
echo "Extracted PlateID: $plateID"
curl -L --retry 5 --continue-at - --progress-bar -o "$plateID" "$url" || echo "failed for PlateID: $url"

