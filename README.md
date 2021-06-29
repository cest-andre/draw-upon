# Draw Upon

In its current state, this is more of a workbench for me to tinker around with neural network models that perform sketch object recognition.  Using the web client (sketch_page.html), the user may create a drawing and submit it to the server (sketch_server.py).  The server loads up a pretrained network (sketch_detect.py) and returns the results.  These results will be presented in a digestable form so that the user can better understand how the network made its prediction.  Details about the project's current state are sketched below.


## Client-side Dependencies

Paper.js  0.12.11    ---  http://paperjs.org/

jQuery    3.5.1      ---  https://jquery.com/

D3        6.7.0      ---  https://d3js.org/

Notes:  D3 provides a src link which is hardcoded in sketch_page.html.  Paper.js and jQuery must be downloaded (again refer to sketch_page.html for where it is referenced).


## Server-side Dependencies

Python         3.7        ---  https://www.python.org/

TensorFlow     2.3.2      ---  https://www.tensorflow.org/

NumPy          1.18.5     ---  https://numpy.org/

opencv-python  4.5.1.48   ---  https://pypi.org/project/opencv-python/


## Supported Categories

ant, banana, bicycle, brain, car,
cat, coffee cup, computer, eye, guitar,
hammer, helicopter, hourglass, house, key,
light bulb, map, onion, pizza, river,
saxophone, shoe, skateboard, tree, wine glass


## Instructions

Run sketch_server.py and fire up sketch_page.html in a web browser (I've used Chrome and Firefox).  Click and drag the mouse to draw (does not support erase now - refresh page to start over) and press the '3' key to ship it off to the server for sketch prediction.  If successful, the server will return data that the page visualizes beneath where you drew your sketch.  Scroll down to explore!


## Summary and Background

Draw Upon is a simple web application which provides a peak under the hood at a deep neural network performing sequential sketch object recognition.  Taking a supervised learning approach, I downloaded thousands of sketches from Google's QuickDraw dataset (https://quickdraw.withgoogle.com/).  I am currently using a dataset consisting of 50,000 examples (25 categories, 2,000 / category).  These sketches were downloaded in sequential vector format (sequenced by strokes in the order it was drawn) but I am particularly interested in sequences of raster (video) data.  Therefore, I used Paper.js to import QuickDraw data into the web client and convert it to a series of raster frames of the sketch being drawn in the order it was originally drawn.

With the help of Keras, I built a few simple neural networks that perform supervised object classification learning on these sketches.  The networks make use of Keras' convolutional long short-term memory (ConvLSTM) layer to process the sequence of images and produce a category prediction.  I have already trained a few models that perform well on my evaluation dataset (80/20 split - train with 1600 / cat, evaluate with 400 / cat).  With the locally hosted server running, I can use the web client to draw my own sketch, convert it into a series of raster frames, and ship it off to the server where a pretrained network makes a prediction of the object category.

The server returns a few things: the transformed input frames (downsampled and color inverted), predictions, and hidden states for each frame, and finally the eval set confusion matrices for the three top performing models.  Additionally, I have trained a feedforward CNN on only completed sketches (last frame of each sequence).  When a sketch is submitted, the individual frames are also fed to this network with its prediction results for each frame presented along side the ConvLSTM's predictions.

Convolutional LSTM Network paper: https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf

See "test-mug.png" for an example of when the web page updates with prediction results.


## Future Plans

A top priority for this project is to incorporate an attention mechanism and subsequently visualize attention dynamics.  I am investigating Mnih et al.'s Glimpse Network for foveated vision.  I am also investigating Mittal et al.'s work in Bidirectional Recurrent Independent Mechanisms for self-attention and top-down feedback.  I will attempt to adopt these ideas and adapt them for the convolutional LSTM where the hidden states are 3D (width, height, channels) rather than a vector.

Once incorporated, I am interested to see if these mechanisms improve out-of-domain generalization in object recognition.  The experiment I have in mind is to train on natural image data on a set of categories, and then test on QuickDraw sketch data of the same categories.  In this case, a sample will be a single image but a sequence will be generated via glimpses over the image (natural image for training and a finished sketch for testing).  Will the model be able to make use of top-down information to fill in the gaps when analyzing sketch data?  Will its selections for foveated saccades fit well with where animals look?  Will generalization performance improve compared to ablated versions of the network?  I'm excited to find out!  

Back to sketch sequences, I'd like to develop an algorithm that determines which frame causes the most significant disturbance in category prediction and hidden states.  This would make for an interesting proxy for which portions of a sketch are most salient.
