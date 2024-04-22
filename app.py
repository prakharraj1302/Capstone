
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


#  additional IMPORTS
import plotly.graph_objects as go
# from sttime import st_timeline
import base64
from shapely.geometry import Point
import datetime
import smtplib
import geopandas as gpd
import pandas as pd
from streamlit_folium import folium_static
from folium.plugins import Search
import folium
from PIL import Image
from streamlit_lottie import st_lottie
import streamlit as st
import requests
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly

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
    path = "{}/{}_model.json".format(selected_model, city)
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

# NEW
def heatwave_prepare(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.resample("d").max()
    df = df.reset_index()
    df["date"] = df["datetime"].dt.date
    df.set_index("date", inplace=True)
    T = (df["temp"] * 9 / 5) + 32
    df["temp"] = T
    R = df["humidity"]
    # Calculating Heat index using heat index chart formula
    hi = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 6.83783 * (10**-3) * (T * T)
        - 5.481717 * (10**-2) * R * R
        + 1.22874 * (10**-3) * T * T * R
        + 8.5282 * (10**-4) * T * R * R
        - 1.99 * (10**-6) * T * T * R * R
    )
    df["heat_index"] = hi
    df["occurence of heat wave"] = df["temp"].apply(
        lambda x: "yes" if x > 128 else "no"
    )
    return df

def aqi_prepare(df):
    df["dt"] = pd.to_datetime(df["dt"])
    df.set_index("dt", inplace=True)
    df = df.resample("d").max()
    df = df.reset_index()
    df["date"] = df["dt"].dt.date
    df.set_index("date", inplace=True)
    return df

# ---- LOAD ASSETS ----

lottie_coding_1 = load_lottieurl(
    "https://assets5.lottiefiles.com/packages/lf20_2cwDXD.json")

lottie_coding_2 = load_lottieurl(
    "https://assets9.lottiefiles.com/packages/lf20_dXP5CGL9ik.json")

# ---- Title SECTION ----
st.set_page_config(page_title="Team Tarang.ai",
                   page_icon=":tada:", layout="wide")

# ---- CSS----
local_css("style/style.css")


st.sidebar.header('Team Tarang.ai')

st.sidebar.subheader('What you want to Predict?')
selected_model = st.sidebar.selectbox('Choose:', ('Heat wave', 'AQI'))
st.sidebar.write('''

''')
cities = ('Bengaluru','Delhi', 'Chennai', 'Lucknow' )
selected_city = st.sidebar.selectbox('Select a city for prediction', cities)

# image = Image.open('images/logo.png')
# st.sidebar.image(image)


st.sidebar.markdown('''
---
Created with ❤️ by [Team Tarang.ai](https://github.com/iamneo-production/00aa9422-7c04-4b7c-975b-6ed887ff7d95).

''')
# ___MAP___ 
retrain_log_path = "./retrain/{}/{}_retrain_log.csv".format(selected_model, selected_city)
df = pd.read_csv(retrain_log_path)


# Unix timestamp in seconds
unix_timestamp = df["last updated date"].iloc[-1]

# Convert Unix timestamp to datetime object
date_time = datetime.datetime.fromtimestamp(unix_timestamp)

year_string = int(date_time.strftime("%Y"))
month_string = int(date_time.strftime("%m"))
date_string = int(date_time.strftime("%d"))


if selected_model == "Heat wave":
    min_date = datetime.date(2012, 1, 1)
    max_date = datetime.date(year_string, month_string, date_string)
else:
    min_date = datetime.date(2020, 12, 2)
    max_date = datetime.date(year_string, month_string, date_string)

d = st.date_input(
    "Choose a date", datetime.date(2023, 1, 1), min_value=min_date, max_value=max_date
)
with st.container():

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        path_ben = "./versioning/weekone/{}/bangalore_temp_csv.csv".format(selected_model)
        path_del = "./versioning/weekone/{}/delhi_temp_csv.csv".format(selected_model)
        path_luc = "./versioning/weekone/{}/lucknow_temp_csv.csv".format(selected_model)
        path_chn = "./versioning/weekone/{}/chennai_temp_csv.csv".format(selected_model)
        
        #  datetime,datetimeEpoch,tempmax,tempmin,temp,feelslikemax,feelslikemin,feelslike,dew,humidity,precip,precipprob,precipcover

        # path_wa = 'versioning/one/{}/1_Warangal_data.csv'.format(selected_model)
        
        df_ben = pd.read_csv(path_ben)
        df_del = pd.read_csv(path_del)
        df_luc = pd.read_csv(path_luc)
        df_chn = pd.read_csv(path_chn)
        # df_wa = pd.read_csv(path_wa)

        if selected_model == "Heat wave":
            df_ben = heatwave_prepare(df_ben)
            df_del = heatwave_prepare(df_del)
            df_luc = heatwave_prepare(df_luc)
            df_chn = heatwave_prepare(df_chn)

            temp_ben = df_ben.loc[d, "temp"]
            temp_del = df_del.loc[d, "temp"]
            temp_luc = df_luc.loc[d, "temp"]
            temp_chn = df_chn.loc[d, "temp"]
            # temp_wa = df_wa.loc[d, 'temp']
            # Select the temperature and heat index value for a particular date and store it in a variable

            heat_index_ben = df_ben.loc[d, "heat_index"]
            heat_index_del = df_del.loc[d, "heat_index"]
            heat_index_luc = df_luc.loc[d, "heat_index"]
            heat_index_chn = df_chn.loc[d, "heat_index"]
            # heat_index_wa = df_wa.loc[d, 'heat_index']
            cities = {
                "city": ["bengaluru", "delhi", "lucknow", "chennai"],
                "Heat Index": [
                    heat_index_ben,
                    heat_index_del,
                    heat_index_luc,
                    heat_index_chn,
                ],
                "Temperature(°F)": [temp_ben, temp_del, temp_luc, temp_chn],
                "latitude": [12.9767936, 28.6517178, 26.8381, 13.0836939],
                "longitude": [77.590082, 77.2219388, 80.9346001, 80.270186],
            }

            # Convert the city data to a GeoDataFrame
            geometry = [
                Point(xy) for xy in zip(cities["longitude"], cities["latitude"])
            ]
            cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry, crs="EPSG:4326")

            # Save the GeoDataFrame to a GeoJSON file
            cities_gdf.to_file("heatwave_cities.geojson", driver="GeoJSON")

            # Load the city data
            cities = gpd.read_file("heatwave_cities.geojson")

            # Create a folium map centered on the India
            m = folium.Map(location=[17.9774221, 79.52881], zoom_start=6)

            # Create a GeoJson layer for the city data
            geojson = folium.GeoJson(
                cities,
                name="City Data",
                tooltip=folium.GeoJsonTooltip(
                    fields=["city", "Heat Index", "Temperature(°F)"],
                    aliases=["City", "Heat Index", "Temperature(°F)"],
                    localize=True,
                ),
            ).add_to(m)

            # Add a search bar to the map
            search = Search(
                layer=geojson,
                geom_type="Point",
                placeholder="Search for a city",
                collapsed=False,
                search_label="city",
            ).add_to(m)

            folium_static(m, width=500, height=500)
        else:
            df_ben = aqi_prepare(df_ben)
            df_del = aqi_prepare(df_del)
            df_luc = aqi_prepare(df_luc)
            df_che = aqi_prepare(df_chn)
            # df_wa = aqi_prepare(df_wa)

            aqi_ben = df_ben.loc[d, "aqi"]
            aqi_del = df_del.loc[d, "aqi"]
            aqi_luc = df_luc.loc[d, "aqi"]
            aqi_chn = df_che.loc[d, "aqi"]
            # aqi_wa = df_wa.loc[d, "aqi"]

            # Select the temperature and heat index value for a particular date and store it in a variable

            cities = {
                'city': ['bengaluru', 'delhi', 'lucknow', 'chennai'],
                'AQI': [aqi_ben, aqi_del, aqi_luc, aqi_chn],
                'latitude': [12.9767936, 28.6517178, 26.8381, 13.0836939],
                'longitude': [77.590082, 77.2219388, 80.9346001, 80.270186]
            }

            # Convert the city data to a GeoDataFrame
            geometry = [
                Point(xy) for xy in zip(cities["longitude"], cities["latitude"])
            ]
            cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry, crs="EPSG:4326")

            # Save the GeoDataFrame to a GeoJSON file
            cities_gdf.to_file("aqi_cities.geojson", driver="GeoJSON")

            # Load the city data
            cities = gpd.read_file("aqi_cities.geojson")

            # Create a folium map centered on the India
            m = folium.Map(location=[17.9774221, 79.52881], zoom_start=6)

            # Create a GeoJson layer for the city data
            geojson = folium.GeoJson(
                cities,
                name="City Data",
                tooltip=folium.GeoJsonTooltip(
                    fields=["city", "AQI"], aliases=["City", "AQI"], localize=True
                ),
            ).add_to(m)

            # Add a search bar to the map
            search = Search(
                layer=geojson,
                geom_type="Point",
                placeholder="Search for a city",
                collapsed=False,
                search_label="city",
            ).add_to(m)

            folium_static(m, width=520, height=520)


# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        # i1 = "/images/logo-capstone.png"
        # image1 = Image.open(i1)
        # st.image("/images/logo-capstone.png")  
        
        # i2 = "/images/vit-logo.png"
        # image2 = Image.open(i2)
        # st.image("/images/vit-logo.png") 
        
        st.title("Capstone Project - Team Tarang.ai")
        st.write("Stay ahead of the heat and breathe easy with Team Tarang.ai")


        if selected_model == "Heat wave":
            # st.write("[Exploratory Data Analysis(EDA)](https://colab.research.google.com/drive/1xH77_KLE3gpmTxGk9-X36Pj6lHee0iBc?usp=sharing#scrollTo=nybHfIsygGzp)")
            # st.write("[Solution Architecture](https://www.craft.do/s/1eTduABsPuFIDX)")
            
            st.write("---")
            name = "{} Prediction".format(selected_model)
            st.title(name)

            st.header("Introduction:")
            st.write("Our research delves into the escalating frequency and severity of heatwaves, a consequence of shifting weather patterns induced by climate change. We focus on three major Indian cities—Bangalore, Chennai, and Delhi NCR—where rapid urbanization exacerbates the urban heat island effect, amplifying the impacts of extreme heat events on public health and urban infrastructure.")
        
            st.header("Research Methodology:")
            st.write("We employed a multi-faceted approach combining advanced machine learning techniques like Gradient Boosting and Random Forest with traditional time series forecasting models such as ARIMA. By analyzing over three decades of historical weather data, we aimed to discern patterns and predict future heatwave occurrences accurately.")

            st.header("Key Findings:")
            st.write("Our analysis revealed that Gradient Boosting outperformed other models in capturing complex temperature fluctuations, indicating its efficacy in predicting heatwave occurrences with high precision. Random Forest also demonstrated commendable performance, showcasing its robustness in handling variance and making reliable predictions.")

            st.header("Implications:")
            st.write("Understanding heatwave trends and predicting future occurrences is crucial for urban planning and public health interventions. Our findings underscore the importance of integrating advanced analytics into climate resilience strategies, enabling policymakers to mitigate the adverse impacts of extreme heat events on vulnerable populations and infrastructure.")

            st.header("Conclusion:")
            st.write("Our research sheds light on the escalating threat of heatwaves in major Indian cities and underscores the significance of proactive measures to enhance climate resilience. By harnessing the power of advanced machine learning techniques, we offer valuable insights for policymakers and urban planners to develop effective strategies for mitigating the impacts of extreme heat events on public health and urban infrastructure.")


        else:
            # st.write("[Exploratory Data Analysis(EDA)](https://colab.research.google.com/drive/1WgV57xtbG05shrxy47Fw59oOmzTZ2yJv?usp=sharing)")
            # st.write("[Solution Architecture](https://www.craft.do/s/1eTduABsPuFIDX)")
            
            name = "{} Prediction".format(selected_model)
            st.title(name)

            st.header("Introduction:")
            st.write("Our study investigates the deteriorating air quality in major Indian cities, particularly focusing on Bangalore, Chennai, and Delhi NCR. Rapid urbanization and industrialization have contributed to soaring pollution levels, posing significant threats to public health and environmental sustainability.")

            st.header("Research Methodology:")
            st.write("We employed a comprehensive approach, utilizing machine learning models like Random Forest and Gradient Boosting alongside traditional time series forecasting methods such as SARIMA. By analyzing three years of air quality data, we aimed to discern trends and predict future AQI variations accurately.")

            st.header("Key Findings:")
            st.write("Our analysis revealed that Random Forest exhibited strong alignment with actual AQI trends, showcasing its capability to interpret and learn from AQI data over time. Gradient Boosting also showed promising results, indicating its efficacy in capturing complex pollution dynamics and making accurate predictions.")

            st.header("Implications:")
            st.write("Understanding AQI trends and predicting future variations is essential for formulating effective pollution control measures and public health interventions. Our findings highlight the value of integrating advanced analytics into environmental management strategies to mitigate the adverse impacts of air pollution on public health and ecological balance.")

            st.header("Conclusion:")
            st.write("Our research underscores the pressing need to address air pollution in major Indian cities and emphasizes the importance of proactive measures to improve air quality and safeguard public health. By leveraging advanced machine learning techniques, we offer valuable insights for policymakers and environmental agencies to develop targeted interventions and policies for mitigating the impacts of air pollution on human health and the environment.")


    with right_column:
        
        # i = "./images/vit-logo.png"
        # image = Image.open(i)
        # st.image(image)        
        st.markdown("<h4 style='text-align: right; color: white;'>In guidance of:</h4>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: right; color: white;'>Dr. Sandip Mal</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: right; color: white;'>Dr. Preetam Suman</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: right; color: white;'>Dr. Sasmita Padhy</h5>", unsafe_allow_html=True)

        if selected_model == 'Heat wave':
                st_lottie(lottie_coding_1, height=300, key="coding")
        else:
            st_lottie(lottie_coding_2, height=300, key="coding")

        



# ---- Introduction ----
# with st.container():
#     st.write("---")
#     left_column, right_column = st.columns(2)
#     with left_column:
#         st.header("Our Vision and Approach")
#         st.write(
#             """
#             Welcome to Team Tarang.ai a home of Heatwave and AQI Prediction Platform! We are here to help you prepare for extreme weather conditions and make informed decisions to protect yourself and your loved ones.

#             Our platform offers a seamless user journey, starting with the homepage where you can select the criteria you want to predict and the city you are interested in. Once you make your selection, our platform provides you with a graphical representation of the selected criteria for the chosen city, giving you a quick overview of the situation.

#             The platform also includes polar plots and maps for analyzing trends and a map feature for visualizing data for selected cities. Our proposed solution architecture is scalable, adaptable, cost-effective, and dynamic due to retraining and versioning, and CI/CD implementation. Our platform offers a smooth and interactive user experience, providing all necessary information and insights about heatwave and AQI prediction for selected cities.
#             """
#         )

#     with right_column:
#         if selected_model == 'Heat wave':
#             st_lottie(lottie_coding_1, height=300, key="coding")
#         else:
#             st_lottie(lottie_coding_2, height=300, key="coding")


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
        st.plotly_chart(fig1)

    else:
        fig1 = plot_plotly(m, forecast)

        fig1.update_layout(
            plot_bgcolor='#7FFFD4',  # set the background color
            paper_bgcolor='#F8F8F8',  # set the background color of the plot area
        )
        st.plotly_chart(fig1)
else:
    info("Info", '''The Graph displays the prediction and actual AQI Reading for the range of the full dataset and for year 2023
    The orange points shows the predicted value and the grey points shows the actual value of AQI.''')
    fig1 = line_plot_plotly(m, forecast, 'markers', selected_model)

    fig1.update_layout(
        plot_bgcolor='#7FFFD4',  # set the background color
        paper_bgcolor='#F8F8F8',  # set the background color of the plot area
    )
    st.plotly_chart(fig1)



