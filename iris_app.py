import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
from PIL import Image
import shap
import numpy as np

###############CODE#####################
loaded_model = joblib.load("./obj/rf_model.pkl")

#Loading images
setosa = Image.open('./static/iris_setosa.jpg')
versicolor = Image.open('./static/iris_versicolor.jpg')
virginica = Image.open('./static/iris_virginica.jpg')


##############STREAMLIT APP##################

#TITLE
st.title("Iris flower species Classification App")
feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']

#PLOTLY FIGURE
HtmlFile = open("./static/scatter_plot.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height = 400, width=1000)

#SELECTED VALUES
st.sidebar.title("Features")

parameter_list = ['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_default_values=['5.2','3.2','4.2','1.2']
parameter_input_values = []

for parameter, parameter_df in zip(parameter_list, parameter_default_values):
	values = st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
	parameter_input_values.append(values)
	
df = pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')

#PREDICTION
if st.button("Click Here to Classify"):
	col1, col2,col3 = st.columns((6,0.5,10.5))
	with col1:
		prediction_class = loaded_model.predict(df)[0]
		prediction_prob = round(loaded_model.predict_proba(df).max() * 100, 1)
		print(prediction_class);
		print(prediction_prob)

		st.markdown(f"<h2 style='text-align: left; color: #4169e1;'> {prediction_class} with probability {prediction_prob} % </h2>", unsafe_allow_html=True)

		if prediction_class == "Iris-setosa":
			st.image(setosa)
		elif prediction_class == "Iris-versicolor":
			st.image(versicolor)
		else:
			st.image(virginica)

	with col3:
		#SHAP VALUES
		st.markdown(f"<h2 style='text-align: left; color: #4169e1;'> Shap values explanation </h2>", unsafe_allow_html=True)
		explainer = shap.TreeExplainer(loaded_model)
		shap_values = explainer.shap_values(df)

		print(shap_values)

		shap_plot_reprs = []

		for i in range(3):
			shap_plot = shap.force_plot(explainer.expected_value[i],
										shap_values[i],
										df,
										feature_names=feature_names,
										out_names=class_names[i])
			shap_plot_reprs.append(shap_plot._repr_html_())

		shap_html_repr = "".join(shap_plot_reprs)

		components.html(f"<head>{shap.getjs()}</head><body>{shap_html_repr}</body>", height=450)


