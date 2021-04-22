# Draw Upon

Core source code for the Draw Upon prototype.  In its current state, it is more of a workbench for me to tinker around with neural network models that perform sketch object recognition.  Long term ambitions are to convert this into an interactive educational tool for people to learn about the inner workings of artificial neural networks through their own drawings.  Using the web client (sketch_page.html), the user may create a drawing and submit it to the server (sketch_server.py).  The server loads up a pretrained network (sketch_detect.py) and returns the feedforward results.  These results will be presented in a digestable form so that the user can better understand how the network made its prediction.

Details about the project's current state are sketched below.


Client-side dependencies:

Paper.js  0.12.11    ---  http://paperjs.org/

jQuery    3.5.1      ---  https://jquery.com/

D3        6.7.0      ---  https://d3js.org/

Notes:  D3 provides a src link which is hardcoded in sketch_page.html.  Paper.js and jQuery must be downloaded (again refer to sketch_page.html for where it is referenced).

Server-side dependencies:

Python         3.7        ---  https://www.python.org/

TensorFlow     2.3.2      ---  https://www.tensorflow.org/

NumPy          1.18.5     ---  https://numpy.org/

opencv-python  4.5.1.48   ---  https://pypi.org/project/opencv-python/


Summary of the project's current state:

Draw Upon is a simple web application which provides a peak under the hood at deep neural network performing sketch object recognition.  Taking a supervised learning approach, I downloaded thousands of sketches from Google's QuickDraw dataset (https://quickdraw.withgoogle.com/).  I am currently using a subset of data consisting of 50,000 examples (25 categories, 2,000 / category).  These sketches are in sequential vector format (sequenced by strokes in the order it was drawn) but I am particularly interested in sequences of raster (video) data.  Therefore, I used Paper.js to import QuickDraw data into the web client and convert it to a series of raster frames of the sketch being drawn in the same order.

With the help of Keras, I built a few simple neural networks that perform supervised learning on these sketches.  The networks make use of Keras' convolutional long short-term memory (ConvLSTM) layer to process the sequence of images and produce a category prediction.  I have already trained a few models that perform well on my evaluation dataset.  With the locally hosted server running, I can use the web client to draw my own sketch, convert it into a series of raster frames, and ship it off to the server where a pretrained network makes a prediction.  The server returns quite a few things: the predictions and hiddent states for each frame.  I am interested in the network's frame-by-frame confidence and hidden state dynamics to see which parts of the sketch caused significant disturbances.  A top priority for this project is to incorporate an attention mechanism and subsequently visualize attention dynamics.
