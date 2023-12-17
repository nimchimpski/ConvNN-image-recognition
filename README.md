Builds Keras Tensorflow models, with easily configured build options.
This was made primarily to be trained on the gtsrb (German Traffic Sign) dataset, and the parameters are currently optimised for that. 
Seperate testing on random signs shows good generalisation for the current model - with some overgeneralisation (identifying 'stop' text written on road surface as an actual stop sign)

2nd command line input specifies the training data filepath/name, and optional 3rd argument to save the model by giving a filepath/name.

A combined PIL image will be made of:

-chosen parameters
-loss and accuracy plots
-total time
-3d layered representation of the model

...and stored in a 'results' directory.

The 3d visualisation package can be installed with pip:
pip install git+https://github.com/nimchimpski@visualkeras_mod

Original version at https://github.com/paulgavrikov/visualkeras

Video demonstration at https://youtu.be/MYlqsVVgAks


