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
+-- crossvalidation.py             # implementation of 10-fold cross-validation and location-based cross-validation
| ...
+-- preprocessing.py               # implementation of image preprocessing using OpenCV
+-- results.ipynb                  # notebook used to generate comparisons graphs between ResNet-50, SqueezeNet, and VisionFeaturePrint_Scene
|
```

#### Configuration

In `classifier_crossval.ipynb` cell 4, set `BASE_MODEL` variable to desired string value out of 
- `resnet-50`
- `squeezenet_v1.1`
- `VisionFeaturePrint_Scene`

Set `preprocessing` to `True` or `False` to apply or not apply, respectively, preprocessing to images.

Set `final` to `True` to export model to `.mlmodel` format.
