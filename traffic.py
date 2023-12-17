import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import uuid
import datetime
import pydot    
import graphviz
import visualkeras_mod
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from PIL import ImageFont, Image

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
if sys.argv[1] == 'gtsrb-small':
    NUM_CATEGORIES = 3
TEST_SIZE = 0.4
VALIDATION_SPLIT = 0.2

filetime=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# TEST PARAMETERS
NUM_CONVa = 1 #  input/first layer combo
NUM_FILTERSa = 32 # 32, 64, 128
SIZE_KERNELa = (3,3) # 3,3 5,5
STRIDEa = 1
POOLa = True #  add pooling layer
SIZE_POOLa = (2,2) # 2,2 3,3

NUM_CONVb = 1 #  additonal convolutional layers
NUM_FILTERSb = 128 # 32, 64, 128
SIZE_KERNELb = (5,5) # 3,3 5,5
STRIDEb = 1
POOLb = True #  add pooling layer
SIZE_POOLb = (3,3) # 2,2 3,3

SIZE_HIDDENa =  256 # 512 256 128 
NUM_HIDDEN = 1 # 
SIZE_HIDDENb =  128 # 512 256 128 (decreasing usually)
OPTIMIZER = "adam"
DROPOUT = 0.5
INFO = 'dropout just before output layer\nactivation = Tanh, then relu'
# OPTIMIZER = 'RMSprop'
# OPTIMIZER = 'SGD'
LOSS = "categorical_crossentropy"

# param_grid = {
#     'NUM_CONV': [2, 3],
#     'NUM_POOL': [1, 2],
#     'SIZE_POOL': [(2, 2), (3, 3)],
#     'NUM_FILTERS': [32, 64],
#     'SIZE_KERNEL': [(3, 3), (5, 5)],
#     'NUM_HIDDEN': [1, 2],
#     'SIZE_HIDDEN': [128, 256]
# }

def main():


    #       Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    #       Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    # print(labels)

    #       Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    # print(f"---labels[0] shape={labels[0].shape}")
    # print(f"---labels[0]={labels[0]}")

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    #       Get a compiled neural network
    model = get_model()
    # print(f"---model optimizer={model.optimizer}")

    #       Fit model on training data
    starttime = time.time()

    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
    endtime = time.time()
    totaltime = endtime - starttime

    

    #       Evaluate neural network performance
    print(f"---evaluating model---")
    loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)

    #       create chart
    chartpath = plot(history, totaltime, loss, accuracy)

    #       Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]+'.keras'
        # print(f'---filename={filename}')
        savepath = os.path.join('models', filename)
        model.save(savepath)
        # print(f"Model saved to 'models/{filename}'.")

    fontpath = os.path.join('System','Library','Fonts','HelveticaNeue.ttc')
    font = ImageFont.truetype(fontpath, 20)
    # print(f"---font type---{type(font)}")
    # color_map = {tf.keras.layers.Conv2D: {'fill': '#ff0000', 'outline': 'black'},tf.keras.layers.Dense: {'fill': '#00ff00', 'outline': 'black'},tf.keras.layers.Dropout: {'fill': '#0000ff', 'outline': 'black'},tf.keras.layers.MaxPooling2D: {'fill': '#ffff00', 'outline': 'black'},tf.keras.layers.Flatten: {'fill': 'khaki', 'outline': 'black'}}
    color_map = {tf.keras.layers.Conv2D: {'fill': '#FF0000', 'outline': 'black'}, tf.keras.layers.MaxPooling2D: {'fill': 'lightgray', 'outline': 'black'}, tf.keras.layers.Flatten: {'fill': 'white', 'outline': 'black'}, tf.keras.layers.Dropout: {'fill': 'darkgray', 'outline': 'black'}, tf.keras.layers.Dense: {'fill': 'blue', 'outline': 'black'}}

    # print(f'---Conv2D color={color_map[tf.keras.layers.Conv2D]}')

    #     draw and show visualkeras_mod
    layeredpath = os.path.join('plots', f'layered-{filetime}.png')
    visualkeras_mod.layered_view(model,legend=True, color_map=color_map, font=font,  shade_step = 50, spacing=50, one_dim_orientation='x', to_file=layeredpath)

    # model.summary(

    #       save model summary to file
    # sumpath = os.path.join('results', f'model_summary{filetime}.txt')
    # with open(sumpath, 'w') as f:
        # model.summary(print_fn=lambda x: f.write(x + '\n'))

    #     draw and save model flow
    flow_path = os.path.join('results', f'flow-{filetime}.png')
    plot_model(model, to_file=flow_path , show_shapes=True, show_layer_names=True)

    combine_images(chartpath, layeredpath)
    

def plot(history, totaltime, loss, accuracy):

    epochs = range(1, EPOCHS + 1)
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    plt.figure(figsize=(12, 8))
    # Background color for the entire figure
    plt.gcf().set_facecolor('black')
    # Plot training & validation accuracy values
    
    # first subplot ////////////////////////////////
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', color='white')
    plt.xlabel('Epoch',color='white')
    plt.ylabel('Accuracy',color='white')
    legend = plt.legend(facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    plt.tick_params(colors='white')  # Changing tick color to white
    plt.gca().spines['bottom'].set_color('white')  # Change x-axis line color to white
    plt.gca().spines['left'].set_color('white')    # Change y-axis line color to white  

 


    # second subplot ////////////////////////////////
    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_loss, 'bo-', label='Training Loss', )
    plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss', )
    plt.title('Training and Validation Loss', color='white')
    plt.xlabel('Epoch', color='white')
    plt.ylabel('Loss', color='white')
    legend = plt.legend(facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    plt.tick_params(colors='white')  # Changing tick color to white
    plt.gca().spines['bottom'].set_color('white')  # Change x-axis line color to white
    plt.gca().spines['left'].set_color('white')    # Change y-axis line color to white


  


    # third subplot ////////////////////////////////
    plt.subplot(2, 2, 3)
    # create parameters
    parameters = f'CONFIGURATION:\n\nDATA: {sys.argv[1]}   LOSS: {LOSS}   OPTIMIZER: {OPTIMIZER}\n\nNUM_CONVa:{NUM_CONVa}   NUM_FILTERSa:{NUM_FILTERSa}   SIZE_KERNELa:{SIZE_KERNELa}\nSTRIDEa:{STRIDEa}   POOLa:{POOLa}   SIZE_POOLa:{SIZE_POOLa}\n\nNUM_CONVb:{NUM_CONVb}   NUM_FILTERSb:{NUM_FILTERSb}   SIZE_KERNELb:{SIZE_KERNELb}\nSTRIDEb:{STRIDEb}   POOLb:{POOLb}   SIZE_POOLB:{SIZE_POOLb}\n\nNUM_HIDDEN:{NUM_HIDDEN + 1}   SIZE_HIDDENa :{SIZE_HIDDENa}   SIZE_HIDDENb :{SIZE_HIDDENb}\nDROPOUT: {DROPOUT}   INFO :{INFO}'


    # draw parameters
    plt.text(0, 0.75, parameters, ha = 'left', va = 'top' ,color='yellow', bbox=dict(boxstyle="round,pad=0.3", fc='black', ec="yellow", lw=1))




    # fourth subplot ////////////////////////////////
    plt.subplot(2, 2, 4)
    
   # Data for the pie chart
    maxtime = 120
    if totaltime > 120:
        titletext = '>120'
        totaltime = 120
        remainder = 0
    else:
        remainder = maxtime - totaltime
        titletext= '{:.2f}'.format(totaltime)
    sizes = [remainder, totaltime]
    plt.pie(sizes, colors=('black', 'yellow'),  shadow=True, startangle=90,textprops={'color':"yellow", 'size': 'x-large', 'ha':'center', 'va':'center'}, wedgeprops = {"edgecolor": "yellow", 'linewidth': 1, 'linestyle': '-'}, radius=0.5)

    plt.title(titletext, color='yellow', fontsize=20, y=0.7)
    plt.text(0,-.7,'Time in seconds ', ha='center', va='center', color='yellow', fontsize=10, )

      # Set the background color for each subplot (axes)
    for ax in plt.gcf().get_axes():
        ax.set_facecolor('black')

        # create results ////////////////////////////////
    accuracy = "{:.4f}".format(accuracy)
    loss = "{:.4f}".format(loss)
    totaltimestr = "{:.2f}".format(totaltime)
    results = f'ACCURACY: {accuracy}\n         LOSS: {loss}'
    # draw results
    plt.figtext(0.5, 0.29, results, ha = 'left', va = 'top' , color='white', bbox=dict(boxstyle="round,pad=0.3", fc='black', ec="white", lw=1), fontsize=15)


    plt.tight_layout()
    # plt.show()
    
    chartname = f"{sys.argv[1]}_{filetime}.png"
    plt.savefig(f'plots/{chartname}')
    # plt.show()
    chartpath = os.path.join('plots', chartname)
    return chartpath

def combine_images(image1, image2):
   
    # Open the images
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    
    # Resize images to the same height, if necessary
    image1_height = image1.size[1]
    image2_height = image2.size[1]  
    image1_width = image1.size[0]
    image2_width = image2.size[0]
    # image2 = image2.resize((int(image2.width * image1_height / image2.height), image1_height))

    # Combine images horizontally
    total_width = max(image1_width, image2.width)
    total_height = image1_height + image2.height
    combined_image = Image.new('RGB', (total_width, total_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (0, image1_height))

    # Save the combined image
    results_path = os.path.join('results', f'results_{filetime}.png')
    combined_image.save(results_path)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for foldername in os.listdir(data_dir):
        # print(foldername)
        folderpath = os.path.join(data_dir, foldername)
        # print(folderpath) 
        if os.path.isdir(folderpath):  
            for imgname in os.listdir(folderpath):
                # print(imgname)
                img = cv2.imread(os.path.join(folderpath, imgname))

                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

                img = img/255.0

                images.append(img)

                labels.append(int(foldername))
    # print(f"---images[0] shape={images[0].shape}")
    # print(len(labels))
    
    return (images,labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()
    # print(f"---num cats= {NUM_CATEGORIES}")
    # print(f"---img width= {IMG_WIDTH}")

    # input/first layer as a convolutional layer and more
    for i in range(NUM_CONVa):
        model.add(tf.keras.layers.Conv2D(NUM_FILTERSa, SIZE_KERNELa, strides=STRIDEa, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation="relu", ))

    if POOLa:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(SIZE_POOLa)))

    # add convolutional layers
    for i in range(NUM_CONVb):
        model.add(tf.keras.layers.Conv2D(NUM_FILTERSb, SIZE_KERNELb, strides=STRIDEb, activation="relu"))

    # model.add(tf.keras.layers.Dense(8,input_shape =(IMG_WIDTH, IMG_HEIGHT, 3), activation = "relu"))

    # Max-pooling layer, pool size ???
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(SIZE_POOL)))

    # add NUM_POOL pooling layers
    if POOLb:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(SIZE_POOLb)))

    # Flatten the output from the previous layer
    model.add(tf.keras.layers.Flatten())        

    # Add a hidden layer + dropout
    model.add(tf.keras.layers.Dense(SIZE_HIDDENa,activation = "relu"))
    

    # add NUM_HIDDEN-1 layers
    for i in range(NUM_HIDDEN-1):
        model.add(tf.keras.layers.Dense(SIZE_HIDDENb,activation = "relu"))

    # Add a dropout layer
    model.add(tf.keras.layers.Dropout(DROPOUT))

    # Add output layer with 'NUM CATEGORIES' units, with sigmoid activation
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES,  activation="softmax"))

    model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
