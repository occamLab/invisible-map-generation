# Tests to run

- Seeing how many tags we need to create an accurate map?
- How various things that vary affect the overall mapping
    - Adding noise, etc.
    - Varying Y on odometry nodes
    - Whether having gravity nodes helps or not. 
    - etc.

# TODOs:

7/11

Richard:
Documentation + validate_analogous_params

Rucha:
Getting delta + other metrics to show up easily without having to run everything

Ayush:
Connecting IDs to tags

7/12

- Reprint tags and remap.
- validate_analogous_params



Hello Matt! 

Thanks so much for making RTABmap; it's been a lifesaver so far. We just wanted to ask a quick question about the coordinate frames RTABmap uses to define its tag/phone poses. 

The project we're currently working on uses the phone's default pose (y -> gravity, z -> towards the user, x -> to the right), so we've experimentally determined quaternions to convert from RTABmap's coordinate frame to ours. However, we're having a lot of trouble mathematically justifying how it works, and we feel like knowing this information would be really helpful. 

7/13

WeightSpecifier: -w flag allows for different weights for odom/tag/tag_sba. Yells when set to negative values, yet duncan appears to have set them to negative values himself? Not certain why. 




camera intrinsics, poses

compute corners from pose. 

Tag the tag pose as a 4x4 matrix -> coordinate system transformations, then multiply by the corner vector as a 4x1. 

Each tag detection is referenced to a camera pose

camera intrinsics:
{
    fx,
    fy,
    cx,
    cy
}

If we were to replace the pixels we see with "idealized" pixels thru a transformation (on the board). 



NEXT 3 WEEKS:

Think of each observation of the tag as a prediction of where the tag is located. Smart averaging? 

aprilTagTracker -- swift class to take a look at.
    - How do you do the averaging? 
    - Some simple averaging:
        - kalman filter?
        - regular unweighted average?
        - There's a whole family of approaches
    - graphSLAM // running the invisible map algorithm over and over again, essentially live.
    - we have RTABmap


The visual thing:
- Grabbing key frames as we move around, extracting visual features.
    - we're essentially already doing this with an April
    - we'd just be doing this more often and with less certainty. 
