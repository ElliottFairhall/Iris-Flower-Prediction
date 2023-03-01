import pandas as pd
import os
from sklearn import datasets
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

PAGE_TITLE = "Data Engineer, Educator Analyst and Technology Enthusiast"

PAGE_ICON = ":chart_with_upwards_trend:"

# Set the title and icon of the application
st.set_page_config(page_title = PAGE_TITLE, page_icon = PAGE_ICON, layout="centered")

# Get the current directory and open the css file
current_dir = Path(__file__).parent if "_file_" in locals() else Path.cwd()
iris_image = current_dir / "assets"/ "images" / "Iris.jpg"
css_file = current_dir / "styles" / "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# load iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target
clf = RandomForestClassifier()
clf.fit(X, Y)

# provide title for page
st.markdown("<h1>Simple Iris Flower Prediction</h1>", unsafe_allow_html=True)

st.markdown("---")

#open iris_impage for page
image = Image.open(iris_image)
st.image(image)

# create information related to project outline
st.markdown("""
<p>In this project, we will be creating a machine learning application for predicting the type of iris based on measurements of its petal 
and sepal length and width. Utilising a combination of modules such as sklearn, numpy, and pandas, we will train a model to accurately 
classify Irises into one of three species based on their physical characteristics. The project could prove to be a useful tool for botanists 
and other interested individuals to quickly and accurately identify different types of Irises.<p>
""", unsafe_allow_html=True)

st.markdown("---")

# create information related to input parameters
st.markdown("""
<h2>Input Parameters</h2>
<p>In this machine learning project, the input parameters are based on the various features of the Iris 
flower, such as sepal length and width, and petal length and width. These input parameters are used as inputs to the machine learning model 
to make predictions about the species of the Iris flower based on the input parameters provided. The user can then adjust the values of these 
input parameters to see how it affects the model's prediction.</p>
""", unsafe_allow_html=True)

# create input parameters in form of slicers
sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.4)
sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.4)
petal_length = st.slider("Petal length", 1.0, 6.9, 1.3)
petal_width = st.slider("Petal width", 0.1, 2.5, 0.2)
data = {
    "Sepal Length": sepal_length,
    "Sepal Width": sepal_width,
    "Petal Length": petal_length,
    "Petal Width": petal_width,
}

# create predict button to provide results and further information on prediction and prediction probibility
if st.button("Predict"):
    features = pd.DataFrame(data, index=[0])
    prediction = clf.predict(features)
    prediction_proba = clf.predict_proba(features)
    st.markdown("---")
    # display a title
    st.subheader('Prediction')
    # display information about prediction
    st.markdown("""
    <p>A prediction is defined as a value or label that is output by a machine learning model, based on inputted data. 
    In this case, the prediction is the class label of the Iris flower. Below the machine learning model has determined what
    is most likely to match, based on the input data provided within the input parameters.</p>
    """, unsafe_allow_html=True)
    # display prediction based on target names
    st.write(iris.target_names[prediction])
    st.markdown("---")
    # display a title
    st.subheader('Prediction Probability')
    # display information about prediction probability
    st.markdown("""
    <p>Prediction probability is defined as the likelihood of the model's prediction being correct. It is a value between 0 and 1 that 
    represents the confidence of the model in its prediction. In this case, the prediction probability is the probability of the Iris 
    flower's class label that the model has determined is most likely to match the input data based on input parameters.</p>
    """, unsafe_allow_html=True)
    # display the prediction probability
    st.write(prediction_proba)
    st.markdown("---")
else:
    # provide warning to press the predict button
    st.warning("Please press 'Predict' to interact with the Simple Iris Flower Prediction")