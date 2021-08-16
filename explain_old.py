#Imports
import joblib
import pandas as pd
import shap
from anchor.anchor_tabular import AnchorTabularExplainer
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

#Streamlit
st.set_page_config(initial_sidebar_state='expanded')

numerical_columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
feature_names = numerical_columns

iris = pd.read_csv("./dat/iris.csv")
X = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

#SELECTED VALUES
st.sidebar.title("Features")
class_names = np.unique(iris['Species']).tolist()

parameter_list = ['Sepal length (cm)', 'Sepal Width (cm)', 'Petal length (cm)', 'Petal Width (cm)']
parameter_default_values = ['5.2', '3.2', '4.2', '1.2']
parameter_input_values = []

for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values = st.sidebar.slider(label=parameter, key=parameter, value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input_values.append(values)

user_df = pd.DataFrame([parameter_input_values], columns=parameter_list, dtype=float)
st.write('\n\n')

clf = joblib.load("./obj/rf_model.pkl")
class_prediction = clf.predict(user_df)[0]

#st.sidebar.write(f"## Prediction: {class_prediction}")
#proba = clf.predict_proba(user_df)
#proba_df = pd.DataFrame(proba, columns=class_names).round(3)
#st.sidebar.write(proba_df)

#PREDICTION
st.title("SHAP VALUES AND ANCHORS")
if st.sidebar.button("Click Here to Classify"):
    st.write(f"## Explanation for Predicting: **{class_prediction}**")
    st.subheader("SHAP values")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(user_df)

    print(shap_values)

    shap_plot_reprs = []

    for i in range(3):
        shap_plot = shap.force_plot(explainer.expected_value[i],
                                    shap_values[i],
                                    user_df,
                                    feature_names=feature_names,
                                    out_names=class_names[i])
        shap_plot_reprs.append(shap_plot._repr_html_())

    shap_html_repr = "".join(shap_plot_reprs)

    components.html(f"<head>{shap.getjs()}</head><body>{shap_html_repr}</body>", height=420)

    st.header("Anchors")

    #anchor_explainer = AnchorTabularExplainer(class_names, feature_names, X)

    #exp = anchor_explainer.explain_instance(user_df,clf.predict)
    #exp_html = exp.as_html()
    #components.html(exp_html, height=700)