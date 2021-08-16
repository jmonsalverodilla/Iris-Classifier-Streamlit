# Iris Classifier Streamlit

Simple dashboard for making predictions for the iris dataset including SHAP values.

![](streamlit-iris_app.gif)

<p align="center">
<a href="https://iris-classifier-streamlit-app.herokuapp.com/" target="blank">
    <img align="center" src="https://img.shields.io/badge/Heroku-6762A6?style=for-the-badge&logo=heroku&logoColor=white"/>
</a>  

## Usage

0. Install [anaconda](https://www.anaconda.com/products/individual).

1. Create a virtual environment:

```bash
conda create -n env_iris_streamlit python=3.7
conda activate env_iris_streamlit
```
2. Clone this repository

```bash
git clone https://github.com/jmonsalverodilla/Iris-Classifier-Streamlit.git
cd Iris-Classifier-Streamlit
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run iris_app.py --server.runOnSave True
```

## License

This repo is under the [MIT License](LICENSE).