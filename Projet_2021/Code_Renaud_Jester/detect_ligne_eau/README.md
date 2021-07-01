# Lane detection on swimming competition videos

The aim of this repository is to create a process allowing us to extract an homography matrix automatically from a frame. We can then apply this process to get the homography matrices of all the frames of a video.
Finally, this data are to be used to make a link between the pixel position of a swimmer to its absolute position in the swimming pool.

We use several computer vision technics. 

## Method

The current method is based on the fact that the official pool lanes are always the same color. Hence, we filter by (color keeping only red and yellow) and then apply a Hough Transform on the image.
Then we use clustering to extract the relevant lines from the Hough Transform output. We can then identify where are the three middle lanes. 
The next step consists in identifying interesting points on the lane (like the 5, 15, 25 and 35 meters marks) to compute compute the homography matrix.

## The code

We use line_detection.py to do get a mask on the lanes on an image.
line_detection_video.py allows us to apply the line detection on a video.
lane_feature_identification.py is used to compute identify interesting and known points of a lane on an image. (work on progress) 

Other scripts are utils or tests. 

## Current process

In the process of detecting the key features on a lane.
Next step is to compute the homography matrix of an image and then apply it to a video (probably some post processing will be needed).
