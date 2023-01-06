import logging as dash_logging
from dash import Dash, html, dcc, Input, State, Output, no_update, ALL, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sugar import setup_species, create_network, read_experimental_runs, simulate
from sugar import CONNECTIONS
import pandas as pd

# makes an input box with some help text
def make_input(property_name, help_text, default_value, kwargs={}):
    components = [
        dbc.Label(property_name, style={"font-weight":"bold", "font-size":"0.8em"}),
        dbc.Input(type="number", value=default_value,
                  id={"type":"input", "property":property_name},
                  debounce=True, required=True, **kwargs),
        dbc.FormText(f"{help_text} (default={default_value})", style={"font-size":"0.7em"}),
    ]
    result = html.Div(dbc.Card(dbc.CardBody(components)),
                      id=f"{property_name}_input_card")
    return result

# options = [
#     { "label": "option 1",
#       "value": 1 }, ...
# ]
def make_radio_input(property_name, help_text, options, default_value):
    components = [
        dbc.Label(property_name, style={"font-weight":"bold", "font-size":"0.8em"}),
        dbc.RadioItems(options=options,
                       value=default_value,
                       id={"type":"input", "property":property_name},
                      ),
        dbc.FormText(f"{help_text} (default={default_value})", style={"font-size":"0.7em"}),
    ]
    result = html.Div(dbc.Card(dbc.CardBody(components)),
                      id=f"{property_name}_input_card")
    return result

# setup sugar stuff
sugars, intermediates, catalyst_active, catalyst_dead = setup_species()
experimental_runs, t_span, t_eval = read_experimental_runs("observations.xlsx", sugars)
print("Loaded.")

# make GUI elements
app_title = dcc.Markdown(id="app_title", children="# Sugar Networks")

k_to_j_ratio_input = make_input("k_to_j_ratio", "how much radical is disfavored", 1000.0)
model_type_options = [ {"label":"simple", "value":"simple"},
                       {"label":"complex", "value":"complex"} ]
model_radio_input = make_radio_input("model_type", "complex is much more expensive", model_type_options, "simple")
catalyst_deactivation_input = make_input("catalyst_deactivation", "rate constant", 50)
fit_button = dbc.Button("simulate", color="primary", id="simulate_button", n_clicks=0)
simulation_graphs = dbc.Card(
    dbc.CardBody(
        [
            dbc.Label("Simulation Graphs", style={"font-weight":"bold", "font-size":"1.0em"}),
            dcc.Graph(id="simulation_graph", config=dict(displayModeBar=False))
        ]
    )
)

model_setup_elements = html.Div([
                            dcc.Store(id="parameters_store"),
                            dbc.Row([
                                dbc.Col(k_to_j_ratio_input, width=2),
                                dbc.Col(model_radio_input, width=2),
                                dbc.Col(catalyst_deactivation_input, width=2),
                                dbc.Col(fit_button, width=2)
                            ]),
                            html.Hr(),
                            simulation_graphs,
                            html.Hr()
                        ])

simple_cards, complex_cards, parameter_names = [], [], []
for connection in CONNECTIONS:
    fields = connection.split("_")
    input1 = make_input(f"{connection}_base", "base rate constant", 0.0)
    input2 = make_input(f"{connection}_overall", "overall selectivity", 1.0)
    input3 = make_input(f"{connection}_ablation", "ablation selectivity", 1.0)
    input4 = make_input(f"{connection}_regeneration", "regeneration  selectivity", 1.0)
    inputs = [input1, input2, input3, input4]
    row = dbc.Row([dbc.Col(i, width=2) for i in inputs])
    model_setup_elements.children.append(row)
    simple_cards.append(f"{connection}_overall_input_card")
    complex_cards.append(f"{connection}_ablation_input_card")
    complex_cards.append(f"{connection}_regeneration_input_card")
    parameter_names.extend([f"{connection}_base", f"{connection}_overall", f"{connection}_ablation", f"{connection}_regeneration"])
parameter_names.append("model_type")
parameter_names.append("catalyst_deactivation")

# setup app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(children=[app_title, model_setup_elements],
                      id="root_div", 
                      style={"margin":"5px 10px 5px 10px"})

# display active parameters only
for id in simple_cards:
    @app.callback(
        Output(id, "hidden"),
        Input({"type":"input", "property":"model_type"}, "value"),
    )
    def change_visibility(model_type):
        return model_type == "complex"
for id in complex_cards:
    @app.callback(
        Output(id, "hidden"),
        Input({"type":"input", "property":"model_type"}, "value"),
    )
    def change_visibility(model_type):
        return model_type == "simple"

# update store when parameters change
@app.callback(
    Output("parameters_store", "data"),
    *[ Input({"type":"input", "property":parameter}, "value") for parameter in parameter_names ]
)
def update_parameter(*values):
    parameters_dict = { k:v for k,v in zip(parameter_names, values) }
    return parameters_dict

# re-run simulation when simulate button is pressed
@app.callback(
    Output("simulation_graph", "figure"),
    Input("simulate_button", "n_clicks"),
    State("parameters_store", "data"),
    prevent_initial_call=True
)
def simulate_and_graph(n_clicks, parameters_dict):
    #print("simulating")
    network = create_network(parameters_dict, sugars, intermediates, catalyst_active, catalyst_dead)
    results = simulate(network, catalyst_active, experimental_runs, t_span, t_eval)
    #print("done")
    sugar_abbreviations = [ s.abbreviation for s in sugars ]

    titles = [ f"start={run.starting_sugar.description}" for run in experimental_runs ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.2)
    for i,(experimental_run, concentrations_df) in enumerate(zip(experimental_runs, results)):
        # interleave observed and simulated data
        concentrations_df = concentrations_df[sugar_abbreviations]
        data_rows, labels = [], []
        for j,experimental_observations in enumerate(experimental_run.observations):
            data_rows.append(experimental_observations)
            labels.append(f"exp:{t_eval[j]} s")
            data_rows.append(concentrations_df.iloc[j,:].to_numpy())
            labels.append(f"sim:{t_eval[j]} s")
        results_df = pd.DataFrame(data_rows, index=labels, columns=sugar_abbreviations)

        # create the bar objects
        row = i // 3 + 1
        col = i % 3 + 1
        count = -1
        colors = ["red","blue"] * len(experimental_run.observations)
        for label,df_row in results_df.iterrows():
            count += 1 
            bar = go.Bar(name=label, x=sugar_abbreviations, y=df_row, marker_color=colors[count])
            fig.add_trace(bar, row=row, col=col)
        fig.update_yaxes(range=[0,1], row=row, col=col)
    fig.update_layout(barmode="group", showlegend=False, height=500)
    return fig

# launch app
log = dash_logging.getLogger("werkzeug")
log.setLevel(dash_logging.ERROR)
app.run_server(debug=True, use_reloader=True, port=8050)