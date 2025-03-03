# ett-vid-clipper

This code is used for automatically finding and extracting clips of [Eleven Table Tennis (ETT)](https://elevenvr.com/en/) points from a recording. Clips are extracted in a lossless manner using ffmpeg.

For instance, running this script for the top 3 duration clips in a video will yield the following:
```
Extracted points:
|   point |   start |     end |   duration |
|--------:|--------:|--------:|-----------:|
|     264 | 3533.55 | 3544.2  |      10.64 |
|     355 | 4713.63 | 4723.27 |       9.64 |
|      99 | 1267.3  | 1276.23 |       8.94 |

Creating video clips ...
Done! Created 3 video clips in 37s
```


You might use this script in a number ways, such as creating a video that removes all down-time from a recording. By default, it will find the points in the match and print their beginning and ending time points. This list can be saved to a csv file.

Optionally, you can have it create video clips of points for you. It will produce individual clips of every point, as well as a single video file containing all clips.

You might use this script to automatically filter out the non-action times from the entire recording. Or, you might use this script to find the top 10 clips from a match and automatically produce a video containing only those points.

Overall, this should run pretty fast on modest hardware. For instance, I ran this on a 1.5 hour match with over 350 total points. It analyzed the video, produced an mp4 of every clip, and produced an mp4 of all the clips combined in less than 5 minutes total. It turned a 1.5 hour video into a 39 minute video of pure action.
```
Processing chunk 1 of 9, start:120.0s, end:689.0s ... completed in 20s
Processing chunk 2 of 9, start:689.0s, end:1258.0s ... completed in 20s
Processing chunk 3 of 9, start:1258.0s, end:1827.0s ... completed in 21s
Processing chunk 4 of 9, start:1827.0s, end:2396.0s ... completed in 20s
Processing chunk 5 of 9, start:2396.0s, end:2965.0s ... completed in 20s
Processing chunk 6 of 9, start:2965.0s, end:3534.0s ... completed in 22s
Processing chunk 7 of 9, start:3534.0s, end:4103.0s ... completed in 20s
Processing chunk 8 of 9, start:4103.0s, end:4672.0s ... completed in 21s
Processing chunk 9 of 9, start:4672.0s, end:5240.0s ... completed in 21s
...
Creating video clips ...
Done! Created 379 video clips in 111s
```

There are a number of options you can use to affect the bevhaior discussed below.

## How to use it

To get help and see a description of the commands, run:
```python
python clipper.py -h
```

To get a csv file of all the points in the video and their start/end times, run:
```python
python clipper.py <video-file> --outcsv
```

To get a list of the top-10 longest points in the video and their start/end times, run:
```python
python clipper.py <video-file> --nclips 10 --orderby duration
```

To create a full video of all points without the dead time in-between points, run:
```python
python clipper.py <video-file> --outclips
```

To get a list of the top-10 points (by duration) and produce video clips of them, run:
```python
python clipper.py <video-file> --outclips --nclips 10 --orderby duration --reverse-clips
```
The `reverse-clips` option will reverse the order of clips in the full clip video. So, for instance, you will see the longest duration (best clip) last rather than first.

Other options to consider:
* Use `--starttime <x>` to specify that you only want to gather clips after x seconds. For instance, you probably don't want to include any warmup before a match.
* If things are not working well, try tweaking `--delta` and `--min_centroid` options. These affect the sound detection aspect.


## How it Works

This script finds clips based on the sound of a ball hitting a paddle. It uses [onset detection](librosa.onset.onset_detect), which works well for these sounds. Initially I had a recording of several paddle hits and tried using cross correlation to find other matches in the video, but this had had false positives and false negatives, though it might have worked OK. Onset detection ended up being simply and performed better. Due to memory constraints, this process needed to be performed in chunks.

Once the timestamps are collected for every paddle hit, groups are determined based on gaps between the timestamps; i.e., sounds close together are part of the same point.

Then, ffmpeg is used to clip the timepoints from the main video file. This process is very fast assuming you do not re-encode the video. Unfortunately, if your timestamp falls on an I-frame (and not a keyframe) you must re-encode the video else the video will not display properly. To prevent, this, we find all the keyframes of the video and sync the starting timestamp of our cuts to the keyframe. **If the video has sparse keyframes, this method would be problematic.**

## Other notes

* This method is not perfect (even though it works reasonably well). If the point ends with the ball still bouncing on the table, this method thinks the point is still happening. Or, if someone is hitting a ball in between matches, we'd pick that up as a point as well.
* I would have liked to have a more accurate count of the number of shots, but I could not figure out how to do this cleanly in the amount of time I spent on this project. Without a more complex model, it's tough to disambiguate the ball hitting the paddle vs. the ball hitting the table vs. the ball hitting the floor.
* Re-encoding the video kills the runtime. This is why we find the nearest keyframe to the starting timestamp of a clip. Generating video clips is free if you don't have to re-encode, but that requires having enough keyframes to where you don't need to do it. AI led me astray trying to re-encode small pieces and stitch them together, but that route never panned out :). I think it's basically re-encode everything or nothing, and everything will be painful.