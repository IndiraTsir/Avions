"""
MODEL APP STREAMLIT
Create your first app

1) install streamlit in terminal and create app.py
2) open in IDE
3) imports in your app plus title
4) open terminal and run : streamlit run uber_pickups.py

"""
# IMPORTS relevant libraries (visualization, dashboard, data manipulation)

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import time, datetime
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from io import BytesIO
import streamlit.components.v1 as components


# from streamlit_shap import st_shap
# import shap
# from sklearn.model_selection import train_test_split
# import xgboost


# TITLE ===================================================

# st.title('Airlines by number of Flights')
st.image("logo3.png", width=300, caption='Application done by Simplon.co')
# Introduction, Airline Cancellations Delays, Analyse Distance Vol, Analyse Temps de Départ, Analysis Total, Conclusions = st.tabs(['Introduction', 'Airline Cancellations and Delays','Analyse Distance Vol', 'Analyse Temps de Départ','Analysis Total', 'Conclusions'])

part = st.sidebar.radio(label='Pages', options=('Intro', 'Delays', 'Flight App', 'Geo Analysis', 'Analysis Total', 'Conclusions'))

# DATASET ==============================================================

df = pd.read_csv("airlines.csv")

# PAGE INTRO ==============================================================

if part == 'Intro':
    st.title('Top 10 Airlines by Number of Flights')

    top_10_airlines_names = df.groupby('AIRLINE_ID').size().sort_values(ascending=False).head(10)

    # Sort the DataFrame by number of flights
    top_10_airlines_names = df.groupby('AIRLINE_NAME').size().sort_values(ascending=False).head(10)

    # Take the top 10 values
    top_10_airlines_names = top_10_airlines_names.head(10)

    # Create the bar plot using Plotly
    fig = go.Figure(data=[
        go.Bar(x=top_10_airlines_names.index, y=top_10_airlines_names.values)
    ])

    fig.update_layout(
        xaxis_title='Airline',
        yaxis_title='Number of Flights',
#         title='Top 10 Airlines by Number of Flights'
    )

    # Display the plot in Streamlit app
    st.plotly_chart(fig)
    
# DELAIS MOYENS ==============================================================

    st.title('Arrivals delays')
    
#     fd = df.sort_values('DEST_CITY_NAME', ascending = False)
    fig = px.histogram(data_frame=df, x='ARR_DELAY')
    fig.update_layout(yaxis=dict(categoryorder="category ascending"))
    fig.update_layout(
        xaxis_title='Arrival Delay',
        yaxis_title='Total count',
#         title='Arrival Delay Flights'
    )
    st.plotly_chart(fig, use_container_width=True)


# BOXPLOT ==============================================================
    st.title('Delays median')
    fig1 = px.box(data_frame=df, x='ARR_DELAY')
    fig1.update_layout(
        xaxis_title='Arrival Delay',
        yaxis_title='Boxplot')
    st.plotly_chart(fig1, use_container_width=True)  
    
    st.markdown('''
            ## Initial Data Introduction
            - We can see that flight delays are very skewed.
            - Majority of flights are expected to be early, or on time
            - Although most flights make good time. Their seems to be a tendancy for extreme outliers.       
                    ''')  
    
    
    # PAGE2 bubble ==============================================================

    
    
#     df_cat = df[['mkt_carrier', 'origin', 'dest', 'arr_delay', 'fl_date']].copy(deep=True)
#     df_cat['arr_delay'] = df_cat['arr_delay'] > 0
#     df_cat['month'] = df_cat.fl_date.dt.month
#     if cat == 'Airlines':
#     # %%
#         y =  df.groupby('mkt_carrier')['arr_delay'].mean()
#         x = df.groupby('mkt_carrier')['mkt_carrier'].count()
#         fig1 = px.scatter(data_frame=df, x=y.index, y=y.values, size=x, size_max=(60),
#                         labels={'x': 'Airline', 'y': 'Mean Arrival Delay', 
#                                 'size': 'Number of Flights'},
#                         title='Arrival Delay by Airline')
#         st.plotly_chart(fig1, use_container_width=True) 
        
        
# PAGE2 ==============================================================
# PAGE2 ==============================================================

elif part == 'Delays':
    st.title('Airline Delays')
    # Calculate the count of occurrences for each category
    count_data = df.groupby(['AIRLINE_NAME', 'CANCELLED']).size().reset_index(name='Count')

        # Create the countplot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=count_data[count_data['CANCELLED'] == 0]['AIRLINE_NAME'],
        y=count_data[count_data['CANCELLED'] == 0]['Count'],
        name='Not Cancelled'
    ))

    fig.add_trace(go.Bar(
        x=count_data[count_data['CANCELLED'] == 1]['AIRLINE_NAME'],
        y=count_data[count_data['CANCELLED'] == 1]['Count'],
        name='Cancelled'
    ))

    fig.update_layout(
        xaxis_title='Airline Name',
        yaxis_title='Count',
        title='Airline Cancellations'
    )

    # Display the plot in Streamlit app
    st.plotly_chart(fig)
    
# PAGE2 part 2 ==============================================================
    st.title('Airline Delays per Month')
    # Filter the DataFrame to include only delayed flights
    delayed_df_carrier = df[df['DEP_DELAY'] > 0]


    # Create a separate bar graph for each airline
    month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
    
# SEABORN PAS TOUCHE ====================================================
#     fig1 = px.histogram(data_frame=df, x='MONTH', y='DEP_DELAY')
#     fig1.update_layout(xaxis_title='Months', yaxis_title='Delays')
#     fig1.update_xaxes(type='category')
    
    
#     for airline_name in delayed_df_carrier['AIRLINE_NAME'].unique():
#         airline_data = delayed_df_carrier[delayed_df_carrier['AIRLINE_NAME'] == airline_name]
#         airline_data['MONTH'] = airline_data['MONTH'].map(month_names)

#         plt.figure()
#         sns.barplot(data=airline_data, x='MONTH', y='DEP_DELAY', ci=None)
        
# #         sns.set_theme(style='dark', palette='deep')
#         sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
#         plt.title(f'Sum of Delays - {airline_name}')
#         plt.xlabel('Month')
#         plt.ylabel('Sum of Delays')
#         plt.xticks(rotation=80)
#         st.pyplot(plt.gcf())
        
# SEABORN PAS TOUCHE ====================================================

    # Create a personalized bar graph for each airline
    for airline_name in delayed_df_carrier['AIRLINE_NAME'].unique():
        airline_data = delayed_df_carrier[delayed_df_carrier['AIRLINE_NAME'] == airline_name]
        airline_data['MONTH'] = airline_data['MONTH'].map(month_names)

        fig = go.Figure(data=[go.Bar(
            x=airline_data['MONTH'],
            y=airline_data['DEP_DELAY']
        )])

        fig.update_layout(
            title=f'Number of Delays - {airline_name}',
            xaxis_title='Month',
            yaxis_title='Number of Delays'
        )

        # Display the plot in Streamlit app
        st.plotly_chart(fig)

    
# PAGE3 ==============================================================
# PAGE3 ==============================================================

# Main content
elif part == 'Flight App':
    st.title('Flight App')
#     # AContent of Page 1 here
    
#     # CSS to set the background image with transparency
#     st.markdown(
#         """
#         <style>
#         body {
#             background-image: url('pictures_planes/Airbus_A380-800_Emirates_At_Dubai_Terminal.jpg');
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#             background-color: rgba(255, 255, 255, 0.8); /* Adjust the transparency here */
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Load flight data and airline name data
#     df_data_mini_mini = pd.read_csv("airlines.csv")
#     df_airline_name = pd.read_csv("data/airlines_names_carriers.csv")

#     # Merge flight data with airline name data on AIRLINE_ID
#     df_merged = pd.merge(df_data_mini_mini, df_airline_name, on="CARRIER")

    # Create airline selection dropdown
    airline_list = sorted(df["AIRLINE_NAME"].unique())
    selected_airline = st.selectbox(f":compass: Select your airline", airline_list)

    # Filter data by selected airline
    filtered_data = df[df["AIRLINE_NAME"] == selected_airline]

    # Create departure airport selection dropdown
    departure_airports = sorted(filtered_data["ORIGIN"].unique())
    selected_departure_airport = st.selectbox(f":airplane_departure: Select your departure airport", departure_airports)

    # Filter data by selected departure airport
    filtered_data = filtered_data[filtered_data["ORIGIN"] == selected_departure_airport]

    # Create arrival airport selection dropdown
    arrival_airports = sorted(filtered_data["DEST"].unique())
    selected_arrival_airport = st.selectbox(f":airplane_arriving: Select your arrival airport", arrival_airports)

    # Filter data by selected arrival airport
    filtered_data = filtered_data[filtered_data["DEST"] == selected_arrival_airport]

    # Display filtered flight data
    st.write(filtered_data)




# PAGE4 ==============================================================
# PAGE4 ==============================================================

elif part == 'Geo Analysis':
    st.title('Persepectives')
    # Content of Page 2 

    st.subheader('Tracking flight')

    # Embed Google Earth map
    components.html(
        """
        <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d396196.0038535864!2d-74.0059726300296!3d40.71277579459063!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x89c24fa5d33f083b%3A0xc80b8f06e177fe62!2sNew%20York%2C%20NY%2C%20USA!5e0!3m2!1sen!2sca!4v1585325999776!5m2!1sen!2sca" width="800" height="600" style="border:0;" allowfullscreen="" loading="lazy"></iframe>
        """,
        width=800,
        height=600,
    )
    
    
# @st.experimental_memo
# def load_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
#     d_train = xgboost.DMatrix(X_train, label=y_train)
#     d_test = xgboost.DMatrix(X_test, label=y_test)
#     params = {
#         "eta": 0.01,
#         "objective": "binary:logistic",
#         "subsample": 0.5,
#         "base_score": np.mean(y_train),
#         "eval_metric": "logloss",
#         "n_jobs": -1,
#     }
#     model = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
#     return model


# st.header('Input data')
# X,y = load_data()
# X_display,y_display = shap.datasets.adult(display=True)


# PAGE5 ==============================================================
# PAGE5 ==============================================================

# ! ! ! ! ! LAISSEER EN COMM TROP DE TEMPS§§§§

elif part == 'Analysis Total':
    st.title('Total analysis')

    st.header('streamlit_pandas_profiling')

    df = pd.read_csv("data/random_subset_extramini.csv")

    pr = df.profile_report()
    st_profile_report(pr)

# PAGE6 ==============================================================
# PAGE6 ==============================================================


elif part == 'Conclusions':
    st.title('Next prospects')

    st.subheader('Input CSV')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      st.subheader('DataFrame')
      st.write(df)
      st.subheader('Descriptive Statistics') # descriptive_stats
      st.write(df.describe())
    else:
      st.info('☝️ Upload a CSV file')    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
# Example 2 RANGE SLIDER ===================================================

# st.subheader('Range slider')

# values = st.slider(
#      'Select a range of values',
#      0.0, 100.0, (25.0, 75.0))
# st.write('Values:', values)

# # """The following three arguments 0.0, 100.0, (25.0, 75.0) represents the minimum and maximum values while the last tuple denotes the default values to use as the selected lower (25.0) and upper (75.0) bound values."""


# # LINE CHART ===================================================

# # st.header('Line chart')

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)


# SELECTBOX ===================================================


# st.header('Selectbox')

# option = st.selectbox(
#      'What is your favorite color?',
#      ('Blue', 'Red', 'Green'))

# st.write('Your favorite color is ', option)

#Text is 'What is your favorite color?'
# possible values to select are ('Blue', 'Red', 'Green')"""

# MULTI_SELECTBOX ===================================================

# st.header('Multiselect')

# options = st.multiselect(
#      'What are your favorite colors',
#      ['Green', 'Yellow', 'Red', 'Blue'],)
#      #['Yellow', 'Red']) #appearing when charging

# st.write('You selected:', options)

# COMPONENTS_PANDAS_PROFILING ===================================================

# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# st.header('streamlit_pandas_profiling')

# df = pd.read_csv("data/random_subset_extramini.csv")

# pr = df.profile_report()
# st_profile_report(pr)


# FILE_UPLOAD ===================================================

# st.title('File uploader')

# st.subheader('Input CSV')
# uploaded_file = st.file_uploader("Choose a file")

# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file)
#   st.subheader('DataFrame')
#   st.write(df)
#   st.subheader('Descriptive Statistics') # descriptive_stats
#   st.write(df.describe())
# else:
#   st.info('☝️ Upload a CSV file')



















