#!/bin/bash

SONG_TITLE="next_level"
DATA_DIR="/root/data/$SONG_TITLE"
RESULTS_DIR="/root/results/$SONG_TITLE"

# Create the results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Iterate over all chunk files in the data directory
for CHUNK_PATH in "$DATA_DIR"/chunk*.mp4; do
    CHUNK_NAME=$(basename "$CHUNK_PATH" .mp4)  # e.g., chunk1
    CHUNK_NUMBER=${CHUNK_NAME#chunk}           # Extract only the number

    # Skip if already processed
    if [ -f "$RESULTS_DIR/$CHUNK_NAME.zip" ]; then
        echo "$CHUNK_NAME has already been processed. Skipping."
        continue
    fi

    # Progress output: Start processing
    echo "=== Starting processing of $CHUNK_NAME ==="

    # Run main.py
    python main.py --id 4 --mp4_path "$CHUNK_PATH" --song_title "$SONG_TITLE" --chunk "$CHUNK_NAME"

    # Progress output: main.py execution completed
    echo "Completed execution of main.py for $CHUNK_NAME."

    # Compress the results
    zip -rq "$RESULTS_DIR/$CHUNK_NAME.zip" "$RESULTS_DIR/$CHUNK_NAME"

    # Progress output: Compression completed
    echo "Compressed the result files for $CHUNK_NAME."

    # Delete the original results folder
    rm -r "$RESULTS_DIR/$CHUNK_NAME"

    # Progress output: Original results folder deleted
    echo "Deleted the original results folder for $CHUNK_NAME."

    # Progress output: Processing completed
    echo "=== Completed processing of $CHUNK_NAME ==="
    echo ""
done
