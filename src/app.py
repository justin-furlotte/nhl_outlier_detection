import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from utils.dash_constants import colors, font, color_dict
pd.set_option('expand_frame_repr', False)  # Display full dataframes while printing them

scatter_df = pd.read_csv("../graphing_data/scatter_df.csv")
scatter_df.loc[scatter_df['Predicted goal pace'] < 0, 'Predicted goal pace'] = 0
scatter_df = scatter_df.round(decimals=1)

years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

app = Dash(__name__)
server = app.server

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        },
        children='Outlier Detection in NHL Scoring'
    ),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        },
        children='''
        A mathematical model for finding underperformers/overperformers.
    '''),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        },
        children='''
        Players above the red line scored more goals than the model predict; players who are below the line scored less than predicted.
    '''),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        },
        children='''
        Try clicking on individual players to open their scoring history compared to the model's predictions.
    '''),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        },
        children='''
        Double click on teams to isolate them. Zoom by clicking or highlighting.
    '''),

    html.Div(html.P([html.Br()])),

    dcc.Slider(
        min=2012,
        max=scatter_df['Season'].max(),
        step=1,
        value=2020,
        marks={str(year): str(year) for year in years},
        id='year-slider'
    ),

    html.Div(
        [
            dcc.Graph(
                id='graph-with-slider',
                clickData={'points': [{'hovertext': 'Jakob Chychrun'}]}
            ),
        ],
        style={'display': 'inline-block', 'width': '60%'}
    ),

    html.Div([
        dcc.Graph(id='time-series')
    ], style={'display': 'inline-block', 'width': '39%'})

])


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    selected_df = scatter_df[scatter_df["Season"] == selected_year]

    fig = go.Figure()
    fig = px.scatter(selected_df, x="Predicted goal pace", y="Goal pace",
                     size="Goal pace", hover_name="Player", color="Team",
                     color_discrete_map=color_dict,
                     log_x=False, size_max=15)

    real_goal_max = selected_df["Goal pace"].max()
    pred_goal_max = selected_df["Predicted goal pace"].max()
    goal_max = np.max([real_goal_max, pred_goal_max])

    fig.add_trace(go.Scatter(x=[0.01, goal_max], y=[0.01, goal_max], line_color='#FF0000'))

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        transition_duration=500
    )

    return fig


@app.callback(
    Output('time-series', 'figure'),
    Input('graph-with-slider', 'clickData'))
def update_time_series(clickData):
    player = clickData["points"][0]["hovertext"]

    time_series_df = scatter_df[scatter_df["Player"] == player]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_series_df["Season"].tolist(),
        y=time_series_df["Goal pace"].tolist(),
        name="Actual"  # this sets its legend entry
    ))

    fig.add_trace(go.Scatter(
        x=time_series_df["Season"].tolist(),
        y=time_series_df["Predicted goal pace"].tolist(),
        name="Predicted"
    ))

    fig.add_annotation(x=0.05, y=0.95, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text='<b>{}</b><br>{}'.format("Player:", player))

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Goal Pace",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        transition_duration=500
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)