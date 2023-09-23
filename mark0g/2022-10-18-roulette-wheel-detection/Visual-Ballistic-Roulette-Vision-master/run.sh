#!/usr/bin/env bash
if [ -z "$1" ]
then
    echo "No argument supplied. Provide the video as argument. Example is: video/1_2.mov"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "File not found!"
    exit 1
fi

echo $1 > video_name.txt

python3 video_converter.py $1
# ffmpeg -i $1 -r 25 videos/frames/output_%04d.png
cd matlab
./run_matlab_2.sh $1
cd ..
python3 main.py