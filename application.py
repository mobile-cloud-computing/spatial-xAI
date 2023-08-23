from io import BytesIO
import io
from tkinter import Image
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
#rom tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np
from PIL import Image ##from here added for XAI, remove if error 4 libs
from keras.utils import load_img
from keras.utils import img_to_array
from lime import lime_image
import matplotlib.pyplot as plt
########################Serialization####################
from json import JSONEncoder
import json
import shap
from keras.preprocessing.image import ImageDataGenerator
## IMPORT FOR OCCLUSION
from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback
from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivity
import cv2
from skimage.segmentation import mark_boundaries
import re
from lime import lime_image
import matplotlib.image as mpimg


dir_path = 'dataset-resized/'
test=ImageDataGenerator(rescale=1/255,
                        validation_split=0.2)
test_generator=test.flow_from_directory(dir_path,
                                        target_size=(300,300),
                                        batch_size=32,
                                        class_mode='categorical',
                                        subset='validation',
                                        shuffle=True)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

def get_ClassName(var):
    result1 = None
    if var == 'cardboard':
        result1 = 0
    elif var == 'glass':
        result1 = 1
    elif var == 'metal':
        result1 = 2
    elif var == 'paper':
        result1 = 3
    elif var == 'plastic':
        result1= 4
    elif var == 'trash':
        result1 = 5
    return result1
    
def load_image_by_name(imageName: str):
    basePath = "dataset-resized/"
    img_path = basePath + imageName
    img = load_img(img_path, target_size=(300, 300))
    img = img_to_array(img, dtype=np.uint8)
    img=np.array(img)/255.0
    return img
class ModelExplainerInterface():
    def load_image_by_image_name(self, image_name: str):
        img = load_image_by_name(image_name)
        print(img)
        label_name, label_class = self.get_class_from_name(image_name)
        return img, label_name, label_class

    # def load_image_by_test_data_index(self, generator, test_data_index: int):
    #     imgs = []
    #     img_path = "dataset-resized/"
    #     names = []
    #     for name in test_generator.filenames:
    #         img = load_img(img_path + name, target_size=(300, 300, 3))
    #         img = img_to_array(img, dtype=np.uint8)

    #         img = np.array(img) / 255.0

    #         names.append(name)
    #     imgs.append(img[np.newaxis, ...])

    #     return imgs, names

    def get_class_from_name(self, image_name: str):
        label_name = image_name.split('/')[0]
        label_class = list(labels.keys())[list(labels.values()).index(label_name)]
        return label_name, label_class

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
#########################################################
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic','Trash']
#def upload_model(model: h5py.File):
    # model = tf.keras.applications.MobileNetV2(weights="imagenet")
#    mymodel = tf.keras.models.load_model(model)
#    print("Model loaded")
#    return model
#mymodel = upload_model()
def load_model():
    #model = tf.keras.applications.MobileNetV2(weights="imagenet")
    model= tf.keras.models.load_model('trained_model.h5')
    print("Model loaded")
    model.summary()
    return model
model = load_model()
def predict(image: Image.Image):
    #image = np.asarray(image.resize((224, 224)))[..., :3]####RGB Image -> 224*224
    #image = np.expand_dims(image, 0)
    #image = image / 127.5 - 1.0
    print("predicting...")
    image = np.asarray(image.resize((300, 300)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image/255.0
    p = model.predict(image)
    cName = class_names[(np.argmax(p[0],axis=-1))]
    response = []
    #resp = {}
    prob = str(np.max(p[0], axis=-1))
    response.append(cName)
    response.append(prob)
    return response



# def read_imagefile(file) -> Image.Image:
#     try:
#         image = Image.open(io.BytesIO(file))
#     except UnidentifiedImageError:
#         print("UnidentifiedImageError: Could not identify the image format.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     return image

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
########################### READ MODEL #################################
#def read_model(file) -> h5py.File:
#    mymodel = h5py.File(BytesIO(file))
#    print(type(mymodel))
#    return mymodel
#########################################################################

##from here added for LIME
def explain_lime(image: Image.Image):
    explainer = lime_image.LimeImageExplainer()
    pred = predict(image)
    # Preprocess the input image
    image = np.asarray(image.resize((300, 300)))[..., :3]
    image = image / 127.5 - 1.0

    # Generate explanations
    explanation = explainer.explain_instance(image, model.predict,
                                             top_labels=6, hide_color=0, num_samples=1000)
    # Retrieve and print the top predicted labels
    # print("Top labels " + str(explanation.top_labels))
    # top_T = str(explanation.top_labels)


    # Retrieve the Lime explanation dictionary
    lime_explanation = str(explanation.local_exp)
    # print("LIME explanation: " + str(lime_explanation))

    # Retrieve the segmentation map
    segments = str(explanation.segments)
    # print("Segments: " + str(segments))

    temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)
    # print("Shape of temp_2: " + str(temp_2.shape))
    # print("Shape of mask_2: " + str(mask_2.shape))
   
    # Convert the masked image to a nested list for JSON serialization
    # encodedNumpyData = json.dumps(temp_2, cls=NumpyArrayEncoder)
    # encodedNumpyData = json.dumps(temp_2.tolist()) 
 
    # Convert the masked image to a NumPy array for replotting
    encodedNumpyData = temp_2*255 
    NumPy= encodedNumpyData.tolist()  

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    ax1.set_title('Lime Explanation')  # Set the title for ax1
    ax1.imshow(mark_boundaries(temp_2, mask_2))
    ax1.axis('off')  # Turn off axis labels and ticks

    plt.savefig('lime.png')  # Save the plot as an image
    plt.close()

    lime = Image.open('lime.png')

    # Plot the bar plot for segment importance
    lime_explanation2 = explanation.local_exp
    segments2 = [seg_idx for seg_idx, _ in lime_explanation2[explanation.top_labels[0]]]
    importance_values = [val for _, val in lime_explanation2[explanation.top_labels[0]]]

    bar_plot_array = [segments2, importance_values]

    # Convert bar plot array to an image
    bar_plot_segments, bar_plot_importance = bar_plot_array
    plt.bar(bar_plot_segments, bar_plot_importance)
    plt.xlabel('Segment')
    plt.ylabel('Importance')
    plt.title('Segment Importance')
    plt.savefig('bar_plot.png')  # Save the plot as an image
    plt.close()

    bar_plot_image = Image.open('bar_plot.png')

    # Overlay the segmentation map on the input image
    segments3 = explanation.segments
    segment_overlay = mark_boundaries(np.array(image), segments3)
    # print("Shape of segment_overlay: " + str(segment_overlay.shape))
    # segment_overlay_array = segment_overlay*255 
    # segment_overlay_array= segment_overlay_array.tolist()  

#/////////////////////////////


    segment_overlay_array = (segment_overlay * 255).astype(np.uint8)

# Convert the overlay array to an Image object
    segment_overlay_img = Image.fromarray(segment_overlay_array)

# Display the overlay image
    plt.figure(figsize=(6, 6))
    imgplot = plt.imshow(segment_overlay_img)
    plt.axis('off')
    plt.title('Segmentation Overlay')
    plt.savefig('Segmentation_Overlay.png')  # Save the plot as an image
    plt.close()

# Save the overlay image
    segment__img = Image.open('Segmentation_Overlay.png')



        
   # Prepare data for plotting
    labels = [str(label) for label in explanation.top_labels]

    top_labels_indices = explanation.top_labels
    top_labels_names = [class_names[idx] for idx in top_labels_indices]


    scores = [score for score in range(1, len(explanation.top_labels) + 1)]

    # Plot the bar plot for top labels
    plt.barh(top_labels_names, scores)
    plt.xlabel('Score')
    plt.ylabel('Label')
    plt.title('Top Predicted Labels')
    plt.xticks(rotation=90)
    plt.savefig('top_T.png')  # Save the plot as an image
    plt.close()

    top_T_plot_image = Image.open('top_T.png')
    top_T = top_labels_names

    # classes2=[]
    # prob2=[]
    # print("\n-------------------Individual Probability--------------------------------\n")

    # for i,j in enumerate (pred[0],0):
    #     print(top_T[i].upper(),':',round(j*100,2),'%')
    #     classes2.append(top_T[i])
    #     prob2.append(round(j*100,2))
    
   
    # index2 = np.arange(len(classes2))
    # plt.bar(index2, prob2)
    # lt.xlabel('Labels', fontsize=12)
    # plt.ylabel('Probability', fontsize=12)
    # plt.xticks(index2, classes2, fontsize=12, rotation=20)
    # plt.title('Probability for loaded image')
    # plt.savefig('top_T.png')  # Save the plot as an image
    # plt.close()

    # top_T_plot_image = Image.open('top_T.png')
    

    

   
    return NumPy, top_T, top_T_plot_image, lime_explanation, segments, bar_plot_image, segment_overlay_array,pred ,lime,segment__img
  

class ShapModelExplainer(ModelExplainerInterface):
    def explain_image_by_image_name(self, image_name: str):
        img, label_name, label_class = super(ShapModelExplainer, self).load_image_by_image_name(image_name)
        img = img[np.newaxis, ...]
        shap_values = self.explain_shap(img)
        print(type(label_class))
        self.plot_explanations(shap_values, img)

    def explain_image_by_test_data_index(self, generator, index: int):
        img = self.load_image_by_test_data_index(test_generator, index)

        temp, mask = self.explain_shap(img)
        self.plot_explanations(temp, mask)

    def build_background_for_computing(self, generator):
        background_imgs = []
        for name in generator.filenames:
            img, label_name, label_class = super(ShapModelExplainer, self).load_image_by_image_name(name)
            background_imgs.append(img[np.newaxis, ...])

        return background_imgs

    def explain_shap(self, image: Image.Image,filename):
        #print("Explaining with SHAP")
        image = np.asarray(image.resize((300, 300)))[..., :3]
        image = np.expand_dims(image, 0)
        image = image / 255.0
        p = model.predict(image)

        actual_name = re.sub('[^a-zA-Z]', '', filename)
        actual_name = actual_name.capitalize()

        cName = class_names[(np.argmax(p[0],axis=-1))]
        response = []
        #resp = {}
        prob = str(np.max(p[0], axis=-1))
        response.append(cName)
        response.append(prob)
 
        # DeepExplainer to explain predictions of the model
        background_imgs = self.build_background_for_computing(test_generator)
        explainer = shap.DeepExplainer(model,background_imgs)  # compute shap values
        shap_values = explainer.shap_values(image, check_additivity=False)
        encodedNumpyData = json.dumps(shap_values, cls=NumpyArrayEncoder)
        # self.plot_explanations(shap_values, image)

    # Prepare to augment the plot
        shap.image_plot(shap_values, image, show=False)
        fig = plt.gcf()
        allaxes = fig.get_axes()

         # Show the actual/predicted class
        allaxes[0].set_title('Pred:  {}'.format(prob))

        for x in range(1, len(allaxes)-1):
            proba = p[0][x-1]
            if isinstance(prob, (float, int)):
                allaxes[x].set_title('{:.2%}'.format(proba), fontsize=14)
            else:
                allaxes[x].set_title(str(proba), fontsize=14)

        plt.savefig('shap_V.png')  # Save the plot as an image
        plt.close()

        shap_V_plot_image = Image.open('shap_V.png')
        
        # print(shap_values)
        # Convert shap_values list to numpy array
        shap_values = np.array(shap_values)
        # print(shap_values.shape)
        # print("cName : ", cName, ", prob : ", prob)

        
        return encodedNumpyData ,shap_V_plot_image,response
        # shap_S_plot_image, 

    def plot_explanations(self,shap_values, img):
        shap.image_plot(shap_values, img, labels=list(labels.values()), show=False)
        #plt.title('Shap Explanation')
        plt.show()


class OcclusionSensitityModelExplainer(ModelExplainerInterface):

     def explain_occlusion(self,image: Image.Image,label):
         image = np.asarray(image.resize((300, 300)))[..., :3]
         image = np.expand_dims(image, 0)
         image = image / 255.0
         print("Explaining with Occlusion Sensitivity")
         explained_img_name = 'TESTNAME.png'
         explainer = OcclusionSensitivity()
         data = (image, label)
         grid = explainer.explain(data, model, label, patch_size=15, colormap=cv2.COLORMAP_TURBO)  #
        #  print(grid.shape)
        #  print(grid)
         explainer.save(grid, ".", explained_img_name)
#         self.plot_explanations(explained_img_name)
#         #return grid, explained_img_name
         encodedNumpyData = np.asarray(grid, dtype=np.uint8)
         encodedNumpyData = encodedNumpyData.tolist()
        #  json.dumps(grid, cls=NumpyArrayEncoder)

        #  encodedNumpyData = encodedNumpyData*255 
        #  encodedNumpyData= encodedNumpyData.tolist()  

            # Convert the grid to a numpy array
        #  grid_array = np.asarray(grid, dtype=np.uint8)

         img = mpimg.imread(explained_img_name)
         plt.figure(figsize = (6,6))
         imgplot = plt.imshow(img)
         plt.axis('off')
         plt.title('Occlusion Sensitivity')
         plt.savefig('Occlus_Image.png')  # Save the plot as an image
         plt.close()

         Occlus_Image = Image.open('Occlus_Image.png')

         return encodedNumpyData, Occlus_Image
#
     def plot_explanations(self, img_name):
         img = plt.imread(img_name)
         plt.figure(figsize=(6, 6))
         plt.imshow(img)
         plt.axis('off')
         plt.title('Occlusion Sensitivity')
         plt.show()

