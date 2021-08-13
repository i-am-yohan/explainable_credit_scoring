import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import psycopg2
import pandas as pd
import plotly.io as pio
import argparse
from sqlalchemy import create_engine

#this does all the counterfactual visualization!
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = 'The Main NLP process'
    )

    parser.add_argument(
        'db_user',
        type=str,
        help='The name of the user'
    )

    parser.add_argument(
        'in_password',
        type=str,
        help='The password for the user'
    )

    parser.add_argument(
        'cf_id',
        type=int,
        help='The counterfactual ID'
    )

    args = parser.parse_args()

    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password))

    CF_id = args.cf_id #This is an adjustable parameter
    plot_table = pd.read_sql('''select * from expl.plot_table
                                where cf_id = 'Counterfactual_{}'
                                '''.format(CF_id), engine)

    pio.renderers.default = "browser"

    fig1 = go.Figure(go.Indicator(
        mode = "gauge",
        value = plot_table['case_score'][0],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Counterfactual Score Summary", 'font': {'size': 35}},
        #delta = {'reference': 600, 'increasing': {'color': "RebeccaPurple"}},
        gauge = {
            'axis': {'range': [300, 1000], 'tickwidth': 5, 'tickcolor': "darkblue",'tickfont':{"size":20}},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 600], 'color': 'red'},
                {'range': [600, 700], 'color': 'orange'},
                {'range': [700, 800], 'color': 'lightgreen'},
                {'range': [800, 1000], 'color': 'green'}
                ],
            'threshold': {
                'line': {'color': "black", 'width': 7},
                'thickness': 1,
                'value': plot_table['target_score'][0]
            }}
        ))

    fig1.add_trace(go.Indicator(
        mode="number+delta"
    ,   value=plot_table['target_score'][0]
    ,   title={'text':'Target Score','font' : {"size":25}}
    ,   domain={'x': [0.45, 0.75], 'y': [0.0, 0.30]}
    ,   delta={'reference': plot_table['case_score'][0], 'relative': False}
    ,   number={'font':{'size':60}}
        ))

    fig1.add_trace(go.Indicator(
        mode="number+delta"
    ,   value=plot_table['raw_score'][0]
    ,   delta={'reference': plot_table['case_score'][0], 'relative': False}
    ,   domain={'x': [0.45, 0.75], 'y': [0.25, 0.55]}
    ,   title={
            "text": "Counterfactual <br><span style='font-size:0.75em;color:gray'>{}".format(plot_table['sk_id_curr'][0]),
            'font': {"size": 25}
        }
    ,   number={'font':{'size':60}}
        ))

    fig1.add_trace(go.Indicator(
        mode="number"
    ,   value=plot_table['case_score'][0]
    ,   title={
            "text":"Account <br><span style='font-size:0.75em;color:gray'>{}".format(131594),
            'font' : {"size":35}
            #"text": "Accounts<br><span style='font-size:1em;color:gray'>Subtitle</span><br><span style='font-size:0.8em;color:gray'>Subsubtitle</span>"
            }
    ,   domain={'x': [0.2, 0.55], 'y': [0, 0.4]}
    ,   number={'font':{'size':120}}
        ))

    fig1.show()

    #Counterfactual plots
    CF_Plots = plot_table.loc[plot_table['cf_inclusion_ind'] == True,:]
    cf_feats = CF_Plots.feature.values.tolist()

    fig2 = go.Figure()
    for cf_feat_i in cf_feats:
        CF_Plots_t = CF_Plots.loc[CF_Plots['feature'] == cf_feat_i,['case_value','counterfactual_value']].stack().reset_index().rename(columns = {'level_1':'Attribute',0:cf_feat_i})
        fig2.add_trace(go.Bar(x=CF_Plots_t['Attribute']
                            ,   y=CF_Plots_t[cf_feat_i]
                            ,   marker=dict(color="crimson")
                            ,   name=cf_feat_i
                            ,   text=cf_feat_i
                            #,   color=Counterfactual_Table.reset_index()['index']
                            ,   showlegend=True)
                       )
        #break

    buttons = []
    for i, cf in enumerate(cf_feats):
        visibility = [i==j for j in range(len(cf_feats))]
        button = dict(
                     label = cf,
                     method = 'update',
                     args = [{'visible': visibility},
                        {'title': 'Counterfactual comparison for {}'.format(cf)
                         }
                        ]
        )
        buttons.append(button)

    updatemenus = list([
        dict(active=-1,
             x=-0.15,
             buttons=buttons
        )
    ])

    fig2['layout']['title'] = 'Title'
    fig2['layout']['showlegend'] = False
    fig2['layout']['updatemenus'] = updatemenus

    fig2.show()


#add score waterall????

