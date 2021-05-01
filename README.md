# MEng-Turi-Create
### Turi Create experiments for classifying busyness from TfL JamCam images, including dataset.

Written and tested with Python v3.8.6

#### Directory Tree
```
+-- datasets/             
|    +-- crossval/
|    |   +-- augmented/
|    |   |   +-- ...
|    |   +-- locations/            # images per location
|    |   |   +-- ...
|    |   +-- preprocessed/         # preprocessed images
|    |   |   +-- busy/
|    |   |   +-- not_busy/
|    |   +-- raw/                  # raw images (no preprocessing)
|    |   |   +-- busy/
|    |   |   +-- not_busy/
+-- pickles/                       # saved runs for quicker re-runs
|    +-- ...
+-- classifier_crossval.ipynb      # main notebook; run for 10-fold cross-validation and location-based cross-validation
| ...
+-- results.ipynb                  # notebook used to generate comparisons graphs between ResNet-50, SqueezeNet, and VisionFeaturePrint_Scene
|
```
