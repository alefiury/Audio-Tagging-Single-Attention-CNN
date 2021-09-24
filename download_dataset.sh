#!/bin/bash
DATA_DIR="data"
MESSAGE="Dataset is ready"
if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
    cd "$DATA_DIR"
    wget -c https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip master.zip
    rm master.zip
    cd ..
    echo "$MESSAGE"
else
    echo "$MESSAGE"
fi