
import pandas as pd
from PIL import Image
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
import streamlit as st
import requests
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
# from prophet.holidays import Turkey
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Description
def info(title, text):
    with st.expander(f"{title}"):
        st.write(text)

#to load necessary assests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def conv(x):
    return round(x)

# to load next year prediction
def load_prediction(selected_model, city):
    path = "{}/{}_csv.csv".format(selected_model, city)
    print(path)
    df = pd.read_csv(path)
    return df


# to load model
def load_model(selected_model, city):
    path = "{}/{}_model.csv".format(selected_model, city)
    with open(path, 'r') as fin:
        m = model_from_json(fin.read())  # Load model
    return m

def line_plot_plotly(m, forecast, mode, model):
    past = m.history['y']
    future = forecast['yhat']
    if model == 'AQI':
        future = future.apply(conv)
    timeline = forecast['ds']

    trace1 = go.Scatter(
        x=timeline,
        y=past,
        mode=mode,
        name='Actual',
        line=dict(color='#777777')
    )
    trace2 = go.Scatter(
        x=timeline,
        y=future,
        mode=mode,
        name='Predicted',
        line=dict(color='#FF7F50')
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title='Actual vs. Predicted Values',
        xaxis=dict(title='Date', rangeslider=dict(visible=True),
                   rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )),
        yaxis=dict(title='Value'),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


# ---- LOAD ASSETS ----

lottie_coding_1 = load_lottieurl(
    "https://assets5.lottiefiles.com/packages/lf20_2cwDXD.json")

lottie_coding_2 = load_lottieurl(
    "https://assets9.lottiefiles.com/packages/lf20_dXP5CGL9ik.json")

# ---- Title SECTION ----
st.set_page_config(page_title="Team cl_AI_mate",
                   page_icon=":tada:", layout="wide")

# ---- CSS----
local_css("style/style.css")


st.sidebar.header('Team cl_AI_mate')

st.sidebar.subheader('What you want to Predict?')
selected_model = st.sidebar.selectbox('Choose:', ('Heat wave', 'AQI'))
st.sidebar.write('''

''')
cities = ('Bengaluru')
selected_city = st.sidebar.selectbox('Select a city for prediction', cities)

# image = Image.open('images/logo.png')
# st.sidebar.image(image)


st.sidebar.markdown('''
---
Created with ❤️ by [Team cl_AI_mate](https://github.com/iamneo-production/00aa9422-7c04-4b7c-975b-6ed887ff7d95).

''')


# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Team cl_AI_mate")
        st.write("Stay ahead of the heat and breathe easy with Team cl_AI_mate")
        name = "{} Prediction".format(selected_model)
        st.title(name)
        st.write(
            "Telangana Tier-2 cities - Alidabad, Nizamabad, Karimnagar, Khammam and Warangal."
        )
        if selected_model == "Heat wave":
            st.write("[Exploratory Data Analysis(EDA)](https://colab.research.google.com/drive/1xH77_KLE3gpmTxGk9-X36Pj6lHee0iBc?usp=sharing#scrollTo=nybHfIsygGzp)")
            st.write("[Solution Architecture](https://www.craft.do/s/1eTduABsPuFIDX)")

        else:
            st.write("[Exploratory Data Analysis(EDA)](https://colab.research.google.com/drive/1WgV57xtbG05shrxy47Fw59oOmzTZ2yJv?usp=sharing)")
            st.write("[Solution Architecture](https://www.craft.do/s/1eTduABsPuFIDX)")

    with right_column:
        st.title("")
        # i = 'images/{}_hw2.jpg'.format(selected_model)
        # image = Image.open(i)
        # st.image(image)


# ---- Introduction ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Our Vision and Approach")
        st.write(
            """
            Welcome to Team cl_AI_mate a home of Heatwave and AQI Prediction Platform! We are here to help you prepare for extreme weather conditions and make informed decisions to protect yourself and your loved ones.

            Our platform offers a seamless user journey, starting with the homepage where you can select the criteria you want to predict and the city you are interested in. Once you make your selection, our platform provides you with a graphical representation of the selected criteria for the chosen city, giving you a quick overview of the situation.

            The platform also includes polar plots and maps for analyzing trends and a map feature for visualizing data for selected cities. Our proposed solution architecture is scalable, adaptable, cost-effective, and dynamic due to retraining and versioning, and CI/CD implementation. Our platform offers a smooth and interactive user experience, providing all necessary information and insights about heatwave and AQI prediction for selected cities.
            """
        )

    with right_column:
        if selected_model == 'Heat wave':
            st_lottie(lottie_coding_1, height=300, key="coding")
        else:
            st_lottie(lottie_coding_2, height=300, key="coding")


st.write("---")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Our Model")
st.write("Select the desired criterias from the sidebar")

# ---- Forecast ----
with st.container():

    with st.spinner('Loading Model Into Memory....'):
        m = load_model(selected_model, selected_city)

    forecast = load_prediction(selected_model, selected_city)


path1 = "winner/{}/winner_{}_prediction.csv".format(
    selected_model, selected_city)

st.header("Graph")
if selected_model == 'Heat wave':
    info("Info", '''The Graph displays the forecasted values and their associated uncertainty intervals over time. 
    Shaded areas above and below the line represent the uncertainty interval.
    The blue line represents the forecast prediction.''')

    agree = st.checkbox('Line graph')

    if agree:
        fig1 = line_plot_plotly(m, forecast, 'lines', selected_model)

        fig1.update_layout(
            plot_bgcolor='#7FFFD4',  # set the background color
            paper_bgcolor='#F8F8F8',  # set the background color of the plot area
        )

    else:
        fig1 = plot_plotly(m, forecast)

        fig1.update_layout(
            plot_bgcolor='#7FFFD4',  # set the background color
            paper_bgcolor='#F8F8F8',  # set the background color of the plot area
        )
else:
    info("Info", '''The Graph displays the prediction and actual AQI Reading for the range of the full dataset and for year 2023
    The orange points shows the predicted value and the grey points shows the actual value of AQI.''')
    # fig1 = line_plot_plotly(m, forecast, 'markers', selected_model)

    # fig1.update_layout(
    #     plot_bgcolor='#7FFFD4',  # set the background color
    #     paper_bgcolor='#F8F8F8',  # set the background color of the plot area
    # )


# st.plotly_chart(fig1)

