Computer Vision Package for Microfluidics. For more information, contact akaminer@gatech.edu.

# Commands

## golgi-train
### Description
Train a model on the up to date dataset.

### Arguments
- epochs
How many epochs to train for

- batch
Batch size. Standard is 16

- patience
When to stop training if the model stops improving. Typically 0

- weight_destination
Destination folder for the weights.


## golgi-track
### Description
Run analysis on a video.

### Arguments
- output_folder
The folder that the csv and analysis video will be placed.

- video_path
The path to the video to analyze

- weight_name
The name of the weights to use. See "golgi-weights-list" and "golgi-weights-download"


## golgi-weights-list
### Description
List the weights currently downloaded.


## golgi-weights-download
### Description
Download weights from the huggingface hub.

### Arguments
- repo_id
The huggingface repository id

- model_name
The name of the model

- huggingface_token
The authentication token to provide to huggingface


## golgi-annotate
### Description
Annotate an image and add it to the roboflow dataset.

### Arguments
- annotation_type
Either "auto" or "manual"

- image_path
Path to the image you want to annotate

- resize_constant
Factor to scale image by for improved resolution. Comes at the cost
of approximation on scaling. Typically, use 3

- --model
The name of the model weights to use for auto annotation. Mandatory for 
annotation_type "auto"

- --api_key
Roboflow api key. Either provide it as an argument or provide it after annotating
image
