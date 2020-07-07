# import plotly.express as px
# df = px.data.gapminder().query("year == 2007")
# fig = px.line_geo(df, locations="iso_alpha", continents,projection="mercator")
# fig.show()



import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import re

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

airports = pd.read_csv('airports.csv', index_col="IATA")
routes = pd.read_csv("ftn_edges.csv", thousands=',')


##########################################################################################
##########################################################################################
##########################################################################################
fig = go.Figure()
fig.add_trace(go.Scattergeo(
    locationmode = 'ISO-3',
    lon = airports['Longitude'],
    lat = airports['Latitude'],
    hoverinfo = 'text',
    text = airports['Name'],
    mode = 'markers',
    # marker = dict(
    #     size = 2,
    #     color = 'rgb(255, 0, 0)',
    #     line = dict(
    #         width = 3,
    #         color = 'rgba(68, 68, 68, 0)'
    #     ))
    ))

for i in range(len(routes)):
    fig.add_trace(
        go.Scattergeo(
            locationmode = 'ISO-3',
            lon = [airports.loc[routes['From'][i],'Longitude'], airports.loc[routes['To'][i],'Longitude']],
            lat = [airports.loc[routes['From'][i],'Latitude'], airports.loc[routes['To'][i],'Latitude']],
            mode = 'lines',
            hoverinfo = 'none',
            # alpha = 0.3,
            line = dict(width = 0.1,color = '#550000'),
            # opacity = float(routes['cnt'][i]) / float(df_flight_paths['cnt'].max()),
        )
    )



def update_traces(new_routes):   
    fig.data = []

    fig.add_trace(go.Scattergeo(
    locationmode = 'ISO-3',
    lon = airports['Longitude'],
    lat = airports['Latitude'],
    hoverinfo = 'text',
    text = airports['Name'],
    mode = 'markers+text',
    # marker = dict(
    #     size = 2,
    #     color = 'rgb(255, 0, 0)',
    #     line = dict(
    #         width = 3,
    #         color = 'rgba(68, 68, 68, 0)'
    #     ))
    ))

    for i in range(len(routes)):
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'ISO-3',
                lon = [airports.loc[routes['From'][i],'Longitude'], airports.loc[routes['To'][i],'Longitude']],
                lat = [airports.loc[routes['From'][i],'Latitude'], airports.loc[routes['To'][i],'Latitude']],
                mode = 'lines',
                hoverinfo = 'none',
                # alpha = 0.3,
                line = dict(width = 0,color = '#550000'),
                # opacity = float(routes['cnt'][i]) / float(df_flight_paths['cnt'].max()),
            )
        )

    for i in range(len(new_routes)):
        fig.add_trace(
            go.Scattergeo(
                text=new_routes['From'][i],
                locationmode = 'ISO-3',
                lon = [airports.loc[new_routes['From'][i],'Longitude'], airports.loc[new_routes['To'][i],'Longitude']],
                lat = [airports.loc[new_routes['From'][i],'Latitude'], airports.loc[new_routes['To'][i],'Latitude']],
                mode = 'lines+markers',
                hoverinfo='text',
                # alpha = 0.3,
                line = dict(width = 3,color = '#555500'),
                opacity = 0.3,
            )
        )

        fig.add_trace(go.Scattergeo(
            text=i,
            locationmode = 'ISO-3',
            lon = [airports.loc[new_routes['From'][i],'Longitude']],
            lat = [airports.loc[new_routes['From'][i],'Latitude']],
            hoverinfo='text',
            mode = 'text'
            )
        )

    fig.add_trace(go.Scattergeo(
    locationmode = 'ISO-3',
    lon = [airports.loc[new_routes['From'][0],'Longitude']],
    lat = [airports.loc[new_routes['From'][0],'Latitude']],
    hoverinfo='text',
    text = 0,
    mode = 'text'
            )
        )



    # fig.update_layout(
    #     title_text = 'Post Apocalyptic Air Services',
    #     showlegend = False,
    #     geo = dict(
    #         scope = 'world',
    #         projection_type = 'equirectangular',
    #         showland = True,
    #         # landcolor = 'rgb(243, 243, 243)',
    #         # countrycolor = 'rgb(204, 204, 204)',
    #     ),
    
    return fig

##########################################################################################
######    Algorithms    ##################################################################
##########################################################################################

##########################################################################################
## Floyd's 
##########################################################################################


def floyd(G, source, target):
    num_nodes = len(G.nodes())
    matrix = pd.DataFrame(np.zeros([num_nodes,num_nodes]), columns = G.nodes())
    predecessor = pd.DataFrame(np.zeros([num_nodes,num_nodes]), columns = G.nodes())
    # add column with names
    matrix['Nodes'] = G.nodes()
    matrix = matrix.set_index('Nodes')
    predecessor['Nodes'] = G.nodes()
    predecessor = predecessor.set_index('Nodes')

    for i in G.edges(data = True):
        matrix.loc[i[0],i[1]] = i[2]['weight']
        matrix.loc[i[1],i[0]] = i[2]['weight']
        predecessor.loc[i[0],i[0]] = '^'
        predecessor.loc[i[1],i[1]] = '^'
        predecessor.loc[i[0],i[1]] = '^'
        predecessor.loc[i[1],i[0]] = '^'


    for k in G.nodes():
        for i in G.nodes():
            for j in G.nodes():
                direct = matrix.at[i,j]
                indirect = matrix.at[i,k] + matrix.at[k,j]
                if direct > indirect:
                    predecessor.at[i,j] = k 
                matrix.at[i,j] = min(direct, indirect )
    
    if predecessor.at[source, target] == '^':
        d = {'From':[source], 'To':[target]}
    else:
        d = {'From':[source,predecessor.at[source, target]], 'To':[predecessor.at[source, target], target]}
    return d


##########################################################################################
## Dijkstra's
##########################################################################################
def dijkstra(G, sourceNode, targetNode):
    selectedNodes = nx.dijkstra_path(G, sourceNode, targetNode)
    d = {'From':[], 'To':[]}
    d['From'] = selectedNodes
    d['To'] = selectedNodes[1:]
    d["To"].append(selectedNodes[-1])
    return d

##########################################################################################
## Minimum Spanning Tree
##########################################################################################


##########################################################################################
## Travellipng Salesman Problem
##########################################################################################

def tsp(G,sourceNode):

    num_nodes = len(G.nodes())
    matrix = pd.DataFrame(np.zeros([num_nodes,num_nodes]), columns = G.nodes())
    # add column with names
    matrix['Nodes'] = G.nodes()
    matrix = matrix.set_index('Nodes')


    for i in G.edges(data = True):
        matrix.loc[i[0],i[1]] = i[2]['weight']
        matrix.loc[i[1],i[0]] = i[2]['weight']

    selectedNodes = []

    selectedNodes.append(sourceNode)

    libNodes = list(G.nodes())

    libNodes.remove(sourceNode)
    for j in range(len(libNodes)):
        weight_2 = [sourceNode, 2**32]   
        for i in libNodes:
            weight_1 = [i,matrix.at[selectedNodes[-1], i]]
            if weight_1[1] < weight_2[1]:
                weight_2 = weight_1
                
            else:
                weight_1 = weight_1
        selectedNodes.append(weight_2[0])
        libNodes.remove(weight_2[0])

    d = {'From':[], 'To':[]}
    d['From'] = selectedNodes
    d['To'] = selectedNodes[1:]
    d["To"].append(selectedNodes[0])
    return d



# fig = go.Figure()

# fig.add_trace(go.Scattergeo(
#     locationmode = 'ISO-3',
#     lon = airports['Longitude'],
#     lat = airports['Latitude'],
#     hoverinfo = 'text',
#     text = airports['Name'],
#     mode = 'text',
#     # marker = dict(
#     #     size = 2,
#     #     color = 'rgb(255, 0, 0)',
#     #     line = dict(
#     #         width = 3,
#     #         color = 'rgba(68, 68, 68, 0)'
#     #     ))
#     ))

# for i in range(len(routes)):
#         fig.add_trace(
#             go.Scattergeo(
#                 locationmode = 'ISO-3',
#                 lon = [airports.loc[routes['From'][i],'Longitude'], airports.loc[routes['To'][i],'Longitude']],
#                 lat = [airports.loc[routes['From'][i],'Latitude'], airports.loc[routes['To'][i],'Latitude']],
#                 mode = 'lines',
#                 hoverinfo = 'none',
#                 # alpha = 0.3,
#                 line = dict(width = 0.1,color = '#550000'),
#                 # opacity = float(routes['cnt'][i]) / float(df_flight_paths['cnt'].max()),
#             )
#         )




# def refresh_map(fig, d = d):
    
#     new_routes = pd.DataFrame(data=d)
#     flight_paths = []
#     for i in range(len(routes)):
#         fig.add_trace(
#             go.Scattergeo(
#                 locationmode = 'ISO-3',
#                 lon = [airports.loc[routes['From'][i],'Longitude'], airports.loc[routes['To'][i],'Longitude']],
#                 lat = [airports.loc[routes['From'][i],'Latitude'], airports.loc[routes['To'][i],'Latitude']],
#                 mode = 'lines',
#                 hoverinfo = 'none',
#                 # alpha = 0.3,
#                 line = dict(width = 0.1,color = '#550000'),
#                 # opacity = float(routes['cnt'][i]) / float(df_flight_paths['cnt'].max()),
#             )
#         )

#     for i in range(len(new_routes)):
#         fig.add_trace(
#             go.Scattergeo(
#                 locationmode = 'ISO-3',
#                 lon = [airports.loc[new_routes['From'][i],'Longitude'], airports.loc[new_routes['To'][i],'Longitude']],
#                 lat = [airports.loc[new_routes['From'][i],'Latitude'], airports.loc[new_routes['To'][i],'Latitude']],
#                 mode = 'lines',
#                 hoverinfo='none',
#                 # alpha = 0.3,
#                 line = dict(width = 3,color = '#555500'),
#                 opacity = 0.3,
#             )
#         )


#     fig.update_layout(
#         title_text = 'Post Apocalyptic Air Services',
#         showlegend = False,
#         geo = dict(
#             scope = 'world',
#             projection_type = 'equirectangular',
#             showland = True,
#             # landcolor = 'rgb(243, 243, 243)',
#             # countrycolor = 'rgb(204, 204, 204)',
#         ),
#     )
#     return fig




app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='From',
                options=[{'label': i, 'value': i} for i in airports['Name']],
                value='BOG'
            ),
            dcc.Dropdown(
                id='To',
                options=[{'label': i, 'value': i} for i in airports['Name']],
                value='DEL'
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='Plane Service',
                options=[{'label': i, 'value': i} for i in ['A', 'B', 'C', 'D']],
                value='A'
            ),
            dcc.RadioItems(
                id='Filter',
                options=[{'label': i, 'value': i} for i in ['Cheapest', 'Fastest','World Tour']],
                value='Cheapest',
                labelStyle={'display': 'inline-block'}
            )
        ]
        ,style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    dcc.Graph(id='network-graph')
])


@app.callback(
    Output(component_id='network-graph', component_property='figure'),
    [Input('submit-button-state', 'n_clicks')],
    [State('From', 'value'),
     State('To', 'value'),
     State('Plane Service', 'value'),
     State('Filter','value')
     ])
def update_figure(n_clicks, fr, t, service, filt):
    pattern_IATA = '[A-Z][A-Z][A-Z]'
    source = re.search(pattern_IATA, fr).group()
    target = re.search(pattern_IATA, t).group()
    



    edge_weight = 'Duration {}'.format(service)
    nodes = pd.read_csv('ftn_edges.csv', thousands=',')
    nodes_list = nodes[["From", "To", "{}".format(edge_weight)]].values.tolist()

    G = nx.Graph()
    G.add_weighted_edges_from(nodes_list)
##################################################
##################################################



    if service == 'Cheapest':
        d = floyd(G, source, target)
        new_routes = pd.DataFrame(data=d)
    elif service == 'Fastest':
        d = floyd(G, source, target)
        new_routes = pd.DataFrame(data=d)
    elif service == 'World Tour':
        d = tsp(G, source)
        new_routes = pd.DataFrame(data=d)
    else:
        d = floyd(G, source, target)
        new_routes = pd.DataFrame(data=d)

    
    update_traces(new_routes)


    fig.update_layout(
    title_text = 'Post Apocalyptic Air Services',
    showlegend = False,
    geo = dict(
        scope = 'world',
        projection_type = 'equirectangular',
        showland = True,
        # landcolor = 'rgb(243, 243, 243)',
        # countrycolor = 'rgb(204, 204, 204)',
    ))

    return(fig)




if __name__ == '__main__':
    app.run_server(debug=True)