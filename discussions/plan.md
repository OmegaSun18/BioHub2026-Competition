# Tasks

1. Dataset
- It is stored in geff format.

```
pip install geff
```

- What sort of data is stored?
    It could be images, graphs, or both.

TO DO: Need to look into the data format geff and data format zarr.

2. Starting Point
- Have a look at the following `https://www.kaggle.com/code/jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans`

It has a classical detection method with a lot of jargon and names.

TO DO: Need to look into the detection method and what sort of data patterns are being processed.
TO DO: Object tracking is also required, to a percise degree.

3. Scoring Function
- There is a detailed file on the scoring metric for the copetition.

TO DO: Need to go through and simplify that file for our understanding for future reference.

4. Spying on Competitors

- Check the high score public notebooks and see what they do and don't

## Understanding

### Dataset

Fill in later.

### Problem Overview

The task is to track and link cells in 3d microscopy data.

A simplified version is that there are cells that will undergo mitosis and divide in a 3d space.
We have the microscopy video of this process in which we want to detect and track the cells through
the lifetime of the video.

A brief overview tells us the task is to detect cells in the video, maintain the detections during
cell division, and detect the daughter cells once formed. This must happen throughout the time 
cycle of the video.

The coupled task with this is to perform object tracking to track the cells as they move and divide
through their lifetime. 

Once we have the tracking data, we must create a lineage graph of each cell with its corresponding
state for each timestep of the video. This lineage graph will be the final submission to be scored on.

Take a look here [for more details](./problem.md).

### Scoring Function

Fill in Later