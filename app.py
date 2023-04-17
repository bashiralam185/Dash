from dash import Dash, html, dcc, Input, Output, ctx, dash_table
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import OrderedDict
from dash.dash_table.Format import Format, Scheme, Sign, Symbol
import pickle

# -------------------------------------------- Analyze part -------------------------------------------------
# importing the dataset
bishkek_data = pd.read_csv("assets/Bishkek_data.csv")
data = pd.read_csv("assets/pm2_data.csv")
pollutants = pd.read_csv("assets/grid-export.csv")

#preprocessing of the datasets
data.dropna(inplace=True)
pollutants['Day'] = pd.to_datetime(pollutants['Day'])
data = data.replace('Unhealthy for Sensitive Groups', 'USG')

# ---------------------------------------------Prediction part --------------------------------------------------
# parameter values for 
params = [
    'year', 'month', 'day', 'hour',
    'nowcast', 'aqi', 'raw'
]
Params = [
    'AQI US', 'NO(mcg/m³)', 'NO2(mcg/m³)', 'CH2O(mcg/m³)', 'SO2(mcg/m³)', 'Temperature(°C)','Humidity(%)'
]



# starting of the app
app = Dash(__name__)
server =  app.server


# general layout of the app
app.layout = html.Div([
    html.Div(html.H1( children='Analyze, predict and forecast Air Pollution in Bishkek'), className='Heading'),
    html.Hr(),

    html.Div([
    html.Div(html.Button('Analyze', id='btn-nclicks-1', n_clicks=0), className='top_button'),
    html.Div(html.Button('Predictions', id='btn-nclicks-2', n_clicks=0), className='top_button'),
    html.Div(html.Button('Forecasting', id='btn-nclicks-3', n_clicks=0), className='top_button')],
    className='buttons'),
    html.Br(),
    html.Div(id='container-button-timestamp'),
    # html.Label("Air Pollutants from 2019-2022 in Bishkek"),
    
], className='Main')

# this call back function is repsonsible to display the graphs from the selected categories
@app.callback(
    Output('container-button-timestamp', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('btn-nclicks-2', 'n_clicks'),
    Input('btn-nclicks-3', 'n_clicks'),
)

# this is the main functions
def displayClick(btn1, btn2, btn3):

    # This if statement contains the graphs for analyze purpose
    if "btn-nclicks-1" == ctx.triggered_id:
        graph_1 = px.line(bishkek_data, x="Date", y='median', color='Specie', title='Bishkek City')
        graph_2 = px.pie(data, names='AQI Category', title='Quality of Air in Bishkek from 2019 to 2022')
        # graph_3 = px.bar(data, x="Year", y="AQI",color="AQI Category",  barmode='group', title="Bishkek air pollution per year", width=500, height=400)
        graph_5 = px.line(data, x="Date (LT)", y="AQI",  title='AQI of Bishkek city from 2019 to 2022')
        graph_4 = px.line(pollutants, x="Day", y="PM1(mcg/m³)", title='Concentration of pollutants in Bishkek Air')
        return  html.Div([
            # first graph inside the analyze button
            html.Div([
            html.Div(dcc.Dropdown(
            id='first_dropdown',
            options=['min', 'max', 'median', 'variance'],
            value='median'
            ), className='first_dropdown'),
            html.Div(dcc.Graph(id = 'pollutant',
              figure=graph_1)
              )]), 
            html.Hr(),
            # second graph inside the analyze button
            html.Div([
                html.Div(
                    dcc.Graph(id='pie_chat', 
                              figure=graph_2), className='pie_chart'
                ), 
                html.Div(
                    dcc.Graph(
                        id='bar_chart',
                        figure=graph_5
                    ),className="bar_chart"
                )
            ], className='SecondGraph'),
            html.Hr(),

            #third graph
            html.Div([
            html.Div(
            dcc.Dropdown(
            id='second_dropdown',
            options=['PM1(mcg/m³)', 'PM10(mcg/m³)', 'PM2.5(mcg/m³)', 'NO(mcg/m³)','NO2(mcg/m³)','SO2(mcg/m³)','Temperature(°C)',
                     'Humidity(%)'],
            value='PM1(mcg/m³)'
            ), className='first_dropdown'
            ),
            html.Div(
            dcc.Graph(id='air_pollutants', 
                      figure=graph_4)
            ), ]),

              ], className='analyze')
    
    
    elif "btn-nclicks-2" == ctx.triggered_id:
        return html.Div([
            html.Div([
            html.H3("Predict the Quality of Air"),
            html.Div([
            html.Div(
                dcc.Dropdown(
                id='classifier',
                options=['CatBoost', 'LightGBM', 'KNN'],
                value='Catboost'
            ),className='drop-down'
            ),
            html.Div([
                dash_table.DataTable(
                    id='table-editing-simple',
                    columns=(
                        [{'id': p, 'name': p} for p in params]
                    ),
                    data=[
                        dict(Model=i, **{param: 0 for param in params})
                        for i in range(1)
                    ],
                    editable=True
                ),
                html.Div(
                html.Div(id="example-output"), className='Output_value'
                )],className='Catboost_classifier'),
            html.Div([
                html.Div(id='title'),
                html.Div(id='accuracy'),
                html.Div(id='precision'),
                html.Div(id = 'recall'),
                html.Div(id='f1score'),
            ],className='side_div'),
                
            ], className='First_section_prediction')
            ], className='Prediction-section'),

            html.Div([
            html.H3('Predict the Pollutants in the Air'),
            html.Div([
            html.Div(
                dcc.Dropdown(
                id='pollutants_PM',
                options=['PM1', 'PM2.5', 'PM10'],
                value='PM1'), className='drop-down'
            ),
            html.Div(
                dcc.Dropdown(
                id='reg_models',
                options=['CatBoost_reg', 'Extra Trees', 'Lightgbm', 'Random Forest', 'XGBoost_reg'],
                ), className='drop-down'
            ), 
            html.Div([
                dash_table.DataTable(
                    id='table-editing-simple2',
                    columns=(
                        [{'id': p, 'name': p} for p in Params]
                    ),
                    data=[
                        dict(Model=i, **{param: 0 for param in Params})
                        for i in range(1)
                    ],
                    editable=True
                ),
                html.Div(
                html.Div(id="example-output2"), className='Output_value'
                )],className='Catboost_classifier'),


            html.Div([
                html.Div(id='Title'),
                html.Div(id='r2'),
                html.Div(id='mae'),
                html.Div(id='mse'),

            ], className='side_div'),


            ], className='First_section_prediction')
            ], className='Prediction-section')
        ])



    elif "btn-nclicks-3" == ctx.triggered_id:
        msg = "Button 3 was most recently clicked"


# ---------------------------------------------------Call backs for analyze function ------------------------------------------
# this call back function updates the graphs of pollutants in the analyze section
@app.callback(
            Output('air_pollutants', 'figure'),
            Input('second_dropdown', 'value')
        )
def analyze(Value):
    graph_4 = px.line(pollutants, x="Day", y=Value, title='Concentration of pollutants in Bishkek Air')
    return graph_4


# This callback function updates the first plot in the anaylze sections
@app.callback(
            Output('pollutant', 'figure'),
            Input('first_dropdown', 'value')
        )
def analyze(Value):
    graph_1 = px.line(bishkek_data, x="Date", y=Value, color='Specie', title='Bishkek City')
    return graph_1


# ----------------------------------------- Call backs for predictions ----------------------------------
@app.callback(
    Output('example-output', 'children'),
    Output('title', 'children'),
    Output ('accuracy', 'children' ),
    Output ('precision', 'children' ),
    Output ('recall', 'children' ),
    Output ('f1score', 'children' ),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'),
    # input output for models
    Input('classifier', "value"))


def display_output(rows, columns, value):
    path = 'assets/'+value
    if value =='CatBoost':
        accuracy = '0.995'
        Precision = '0.99'
        Recall = '0.99'
        f1_score = '0.99'
    elif value =="LightGBM":
        accuracy = '0.997'
        Precision = '0.99'
        Recall = '1'
        f1_score = '0.99'
    # elif value == 'XGBoost':
    #     accuracy = '0.98'
    #     Precision = '0.98'
    #     Recall = '0.98'
    #     f1_score = '0.98'
    elif value == 'KNN':
        accuracy = '0.96'
        Precision = '0.97'
        Recall = '0.97'
        f1_score = '0.97'

    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    input_value = df.values[:1]
    pickled_model = pickle.load(open(path, 'rb'))
    pred = pickled_model.predict(input_value)
    print(pred)
    Title = "The evaluation matrices of "+value +" model."
    acc = "Accuracy = " + accuracy
    pre = "Precision = " + Precision
    rec = "Recall  =  " + Recall
    f1 = "F1 Score =  " + f1_score
    pred1 = "The predicted air quality is  " + pred[0] 
    return pred1 ,Title, acc, pre, rec, f1

@app.callback(
    Output('example-output2', 'children'),
    Output('Title', 'children'),
    Output('r2', 'children'),
    Output('mae', 'children'),
    Output('mse', 'children'),
    Input('table-editing-simple2', 'data'),
    Input('table-editing-simple2', 'columns'),
    Input('pollutants_PM', "value"),
    Input('reg_models', 'value')
)

def regression_part(rows, columns, value, value1):
    if value =='PM1':
        model_path = 'assets/PM1/' + value1
        if value1 == 'CatBoost_reg':
            r2_score = 0.85
            MAE = 0.97
            MSE = 2.67
        elif value1 == 'Extra Trees':
            r2_score = 0.85
            MAE = 0.96
            MSE = 2.69
        elif value1 == 'Lightgbm':
            r2_score = 0.81
            MAE = 1.02
            MSE = 3.32
        elif value1 == 'Random Forest':
            r2_score = 0.83
            MAE = 0.98
            MSE = 3.02
        elif value1 == 'XGBoost_reg':
            r2_score = 0.84
            MAE = 1.02
            MSE = 3.32
    if value=='PM2.5':
        model_path = 'assets/PM2.5/' + value1
        if value1 == 'CatBoost_reg':
            r2_score = 0.83
            MAE = 23.62
            MSE = 2020.34
        elif value1 == 'Extra Trees':
            r2_score = 0.85
            MAE = 20.76
            MSE = 1761.79
        elif value1 == 'Lightgbm':
            r2_score = 0.74
            MAE = 31.79
            MSE = 3042.1
        elif value1 == 'Random Forest':
            r2_score = 0.83
            MAE = 22.35
            MSE = 2008.78
        elif value1 == 'XGBoost_reg':
            r2_score = 0.85
            MAE = 21.97
            MSE = 1784.78
    if value =='PM10':
        model_path = 'assets/PM10/' + value1
        if value1 == 'CatBoost_reg':
            r2_score = 0.95
            MAE = 3.03
            MSE = 22.61
        elif value1 == 'Extra Trees':
            r2_score = 0.94
            MAE = 3.18
            MSE = 26.32
        elif value1 == 'Lightgbm':
            r2_score = 0.93
            MAE = 3.67
            MSE = 32.93
        elif value1 == 'Random Forest':
            r2_score = 0.94
            MAE = 3.29
            MSE = 26.39
        elif value1 == 'XGBoost_reg':
            r2_score = 0.93
            MAE = 3.73
            MSE = 32.16

    d = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    input_values = d.values[:1]
   
    pickled_model = pickle.load(open(model_path, 'rb'))
    pred1 = pickled_model.predict(input_values)
    pred = "The predicted value of " + str(value) +" is : " + str(pred1[0])
    title = "The Evaluation Matrices are of the " + value1
    rscore = "R2 score = " + str(r2_score)
    Mean_AE = 'MAE  =  ' + str(MAE)
    Mean_SE = 'MSE  =   '+ str(MSE)

    return pred, title, rscore, Mean_AE, Mean_SE




if __name__ == "__main__":
    app.run_server(debug=True)