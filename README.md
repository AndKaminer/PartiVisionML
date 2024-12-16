Computer Vision Package for Microfluidics. For more information, contact akaminer@gatech.edu.

# Installation

Set up a virtual environment using python's venv functionality. See [this](https://docs.python.org/3/library/venv.html) link. Inside the virtual environment, install the package with pip. Then, you should have access to the commands listed below.
```
    pip install --upgrade golgi-cell-cv
```

# Running HTML Dash App

```
    from golgi import app
    app.run()
```




# Protocols

## Miscellaneous Setup
Upon installing the package, you will have access to the commands listed above.
To get things set up, you will likely want access to a computer vision model.
Reference the huggingface repository to see all of the available weights. Then
use the golgi-download-weights command to download the correct weights. To
check what weights you have available, you can use the golgi-weights-list command.

## Automatic Annotation

Use the golgi-annotate command like this:

```
    golgi-annotate auto {PATH TO IMAGE} -m {MODEL YOU WANT TO USE (SEE golgi-weights-list)} -k {ROBOFLOW API KEY}
```
The roboflow API key can either be supplied at the beginning with the -k argument or 
at the end of the process.

A window will come up with the image you supplied the path to. Use the mouse
to select a window around the cell (doesn't necessarily have to be centered,
but try not to make it too big). Then, press enter.

After a second, the cropped image will come up with the model's best guess of
what the annotation should look like. You can edit this annotation in the same way
you draw an annotation in the standard annotation. When you're finished, press escape.
The image should come up again as a confirmation. Press any key. Then, if you didn't

supply your API key as an argument, you will be asked for your API key.

## Manual Annotation

Use the golgi-annotate command like this:

```
    golgi-annotate manual {PATH TO IMAGE} -k {ROBOFLOW API_KEY}
```

The roboflow API key can either be supplied at the beginning with the -k argument or 
at the end of the process.

A window will come up with the image you supplied the path to. Use the mouse
to select a window around the cell (doesn't necessarily have to be centered,
but try not to make it too big). Then, press enter.

From now on, at any time, you can have access to the following commands:

- Pressing 'm' allows you to toggle a black and white solid view of the contour
- Pressing 'o' allows you to toggle an overlay of the contour on the image

To speed up the process of drawing, you will now choose darkness ranges to get an 
initial estimate of the contour. Basically, you can select a lower and upper bound for
the color that the contour will include. That is, if a pixel is outside of the range
you set for the contour, it will not be included in the contour, and if it is inside,
it will be included. You can change the bounds of this using the 'h', 'j', 'k', and 'l'
keys. It takes a little getting used to, so play around with it!

When you have a rough estimate done, press escape.

Now, you can use the mouse to make precise alterations to the contour, making sure
it lines up with the cell. To toggle between eraser mode and drawing mode, use
the 'c' key. Remember to use the 'm' and 'o' keys to make sure you're correctly
outlining the cell!

When you're done with drawing, press escape.

A window should pop up showing a confirmation of what the outline looks like.
Press escape again. If you didn't provide an API key as an argument, you will be asked
to provide it now.
