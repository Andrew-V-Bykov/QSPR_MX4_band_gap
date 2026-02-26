from typing import Dict, Tuple, Union, Callable, Sequence, Iterable
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression, TweedieRegressor, ElasticNet, HuberRegressor, BayesianRidge
# from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import optuna
import shap
import pickle
import re
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser



######################################################## CODE FOR DATA ANALYSIS ########################################################

visualization_descriptor_names: Dict[str, str] = {'Temperature': 'Temperature',
                                                  'X_t1': 'd(M–X<sub>t1</sub>)',
                                                  'X_t2': 'd(M–X<sub>t2</sub>)',
                                                  'X_d1': 'd(M–X<sub>d1</sub>)',
                                                  'X_d2': 'd(M–X<sub>d2</sub>)',
                                                  'X_d3': 'd(M–X<sub>d3</sub>)',
                                                  'X_d4': 'd(M–X<sub>d4</sub>)',
                                                  't1_t2': '∠ X<sub>t1</sub>–M–X<sub>t2</sub>',
                                                  't1_d1': '∠ X<sub>t1</sub>–M–X<sub>d1</sub>',
                                                  't1_d2': '∠ X<sub>t1</sub>–M–X<sub>d2</sub>',
                                                  't1_d3': '∠ X<sub>t1</sub>–M–X<sub>d3</sub>',
                                                  't1_d4': '∠ X<sub>t1</sub>–M–X<sub>d4</sub>',
                                                  't2_d1': '∠ X<sub>t2</sub>–M–X<sub>d1</sub>',
                                                  't2_d2': '∠ X<sub>t2</sub>–M–X<sub>d2</sub>',
                                                  't2_d3': '∠ X<sub>t2</sub>–M–X<sub>d3</sub>',
                                                  't2_d4': '∠ X<sub>t2</sub>–M–X<sub>d4</sub>',
                                                  'd1_d2': '∠ X<sub>d1</sub>–M–X<sub>d2</sub>',
                                                  'd1_d3': '∠ X<sub>d1</sub>–M–X<sub>d3</sub>',
                                                  'd1_d4': '∠ X<sub>d1</sub>–M–X<sub>d4</sub>',
                                                  'd2_d3': '∠ X<sub>d2</sub>–M–X<sub>d3</sub>',
                                                  'd2_d4': '∠ X<sub>d2</sub>–M–X<sub>d4</sub>',
                                                  'd3_d4': '∠ X<sub>d3</sub>–M–X<sub>d4</sub>',
                                                  'd_average': 'd<sub>average</sub>(M–X)',
                                                  'delta_d': '∆d',
                                                  'sigma_2': 'σ<sup>2</sup>',
                                                  'N_XX': 'N<sub>contacts</sub>(X···X)',
                                                  'N_VdW': 'N<sub>VdW contats</sub>',
                                                  'XX_min': 'd<sub>min</sub>(X···X)',
                                                  'XX_average': 'd<sub>average</sub>(X···X)',
                                                  'VdW_average': 'd<sub>VdW average</sub>(X···X)',
                                                  'XXmin_2r': 'd<sub>min</sub>(X···X)/2·r<sub>VdW</sub>(X)',
                                                  'XXaver_2r': 'd<sub>average</sub>(X···X)/2·r<sub>VdW</sub>(X)',
                                                  'VdW_2r': 'd<sub>VdW average</sub>(X···X)/2·r<sub>VdW</sub>(X)',
                                                  'Bandgap': 'Band gap',
                                                  }



def pair_counts_df(dataframe: pd.DataFrame,
                   column_name_1: str,
                   column_name_2: str
                   ) -> pd.DataFrame:
    '''________________________________________________________________________________________________________________________________
    Compute a contingency table of pairwise value counts for two columns (column_name_1, column_name_2).
    
    Parameters:
        dataframe (pandas.DataFrame): Input DataFrame containing the data.
        column_name_1 (str):  Name of the first column.
        column_name_2 (str):  Name of the second column.

    Return:
        pandas.DataFrame: Contingency table for two columns of DataFrame.
    ________________________________________________________________________________________________________________________________'''
    if column_name_1 not in dataframe.columns or column_name_2 not in dataframe.columns:
        raise KeyError(f"Columns '{column_name_1}' or '{column_name_2}' not found in DataFrame")
    
    result = pd.crosstab(dataframe['M'], dataframe['X'])

    result['Sum'] = result.sum(axis=1)
    result.loc['Sum'] = result.sum(axis=0)

    result = result[sorted(result.columns, key=lambda x: ['I', 'Br I', 'Br', 'Cl Br', 'Cl', 'Sum'].index(x))]
    result = result.sort_index()

    return result



def build_corr_map(dataframe: pd.DataFrame,
                   method: Union[str, Callable] = 'pearson',
                   ) -> pd.DataFrame:
    '''________________________________________________________________________________________________________________________________
    Calculate pair correlations for descriptors.
    
    Parameters:
        dataframe (pandas.DataFrame): Input descriptor matrix.
        method (str | Callable): Method of correlation ("pearson", "kendall", "spearman" or custom function).

    Return:
        pandas.DataFrame: Correlation map.
    ________________________________________________________________________________________________________________________________'''
    corr_map = dataframe.corr(method=method)
    mask = np.tril(np.ones_like(corr_map, dtype=bool), k=-1)
    corr_map = corr_map.where(mask)

    return corr_map



def histogram_by_halogen(dataframe: pd.DataFrame,
                         trace_params: Union[Dict, None] = None,
                         layout_params: Union[Dict, None] = None,
                         ) -> go.Figure:
    '''________________________________________________________________________________________________________________________________
    Draw histogram for band gap values in dataframe with grouping by halogen type.

    Parameters:
        dataframe (pandas.DataFrame): Input dataframe.
        trace_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.
    
    Return:
        go.Figure: Grouped histogram for bandgap value.
    ________________________________________________________________________________________________________________________________'''
    fig = px.histogram(dataframe.replace({'Br I': 'Br and I'}),
                       x='Bandgap',
                       color='X',
                       barmode='overlay',
                       nbins=21,
                       color_discrete_sequence=px.colors.qualitative.T10)
    fig.update_traces(opacity=1,
                      marker=dict(line=dict(color="rgba(0, 0, 0, 1)",
                                            width=1.5)),
                      selector=dict(type="histogram")
                      )
    
    if not isinstance(trace_params, type(None)):
        fig.update_traces(**trace_params)
    
    fig.update_layout(width=1200, height=700,
                      template='simple_white',
                      xaxis=dict(mirror=True,
                                 title=dict(text='Band gap, eV',
                                            font=dict(size=24))),
                      yaxis=dict(mirror=True,
                                 showgrid=False,
                                 title=dict(text='Count',
                                            font=dict(size=28)),
                                 ),
                      legend=dict(x=.775, y=.95,
                                  title=dict(text='X:', side='top center',
                                             font=dict(size=32)),
                                  font=dict(size=32)
                                  ))
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)
    
    return fig



def histogram_by_metal(dataframe: pd.DataFrame,
                       trace_params: Union[Dict, None] = None,
                       layout_params: Union[Dict, None] = None,
                       ) -> go.Figure:
    '''________________________________________________________________________________________________________________________________
    Draw histogram for band gap values in dataframe with grouping by metal type.

    Parameters:
        dataframe (pandas.DataFrame): Input dataframe.
        trace_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.
    
    Return:
        go.Figure: Grouped histogram for bandgap value.
    ________________________________________________________________________________________________________________________________'''
    fig = px.histogram(dataframe.sort_values(by='M'),
                       x='Bandgap',
                       color='M',
                       barmode='overlay',
                       nbins=21,
                       color_discrete_sequence=px.colors.qualitative.Safe[::]
                       )
    fig.update_traces(opacity=1,
                      marker=dict(line=dict(color="rgba(0, 0, 0, 1)",
                                            width=1.5)),
                      selector=dict(type="histogram"))
    
    if not isinstance(trace_params, type(None)):
        fig.update_traces(**trace_params)
    
    fig.update_layout(width=1200, height=700,
                      template='simple_white',
                      xaxis=dict(mirror=True,
                                 title=dict(text='Band gap, eV',
                                            font=dict(size=24))),
                      yaxis=dict(mirror=True,
                                 showgrid=False,
                                 title=dict(text='Count',
                                            font=dict(size=28)),
                                 ),
                      legend=dict(x=.775, y=.95,
                                  title=dict(text='M:',
                                             side='top center',
                                             font=dict(size=32)),
                                  font=dict(size=32)
                                  )
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)
    
    return fig



def double_histograms_by_composition(dataframe: pd.DataFrame,
                                     trace_params: Union[Dict, None] = None,
                                     layout_params: Union[Dict, None] = None,
                                     ) -> go.Figure:
    '''________________________________________________________________________________________________________________________________
    Draw two histogram for band gap values in dataframe with grouping by halogen and metaltype.

    Parameters:
        dataframe (pandas.DataFrame): Input dataframe.
        trace_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.
    
    Return:
        go.Figure: Two grouped histogram for bandgap value.
    _______________________________________________________________________________________________________________________________'''
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=False,
                        vertical_spacing=0.1,)

    for x_val, c in zip(dataframe['X'].unique(), px.colors.qualitative.T10):
        sub_df = dataframe[dataframe['X']==x_val]
        fig.add_trace(go.Histogram(x=sub_df['Bandgap'],
                                   name=str(x_val),
                                   xbins=dict(start=1.6,
                                              end=3.6,
                                              size=0.1),
                                   marker=dict(color=c,
                                             line=dict(color="rgba(0, 0, 0, 1)",
                                                       width=1.5)),
                                   opacity=0.7,
                                   legend='legend1'), 
                      row=1, col=1)


    for m_val, c in zip(dataframe.sort_values(by='M')['M'].unique(), px.colors.qualitative.Safe):
        sub_df = dataframe[dataframe['M']==m_val]
        fig.add_trace(go.Histogram(x=sub_df['Bandgap'],
                                   name=str(m_val),
                                   xbins=dict(start=1.6,
                                              end=3.6,
                                              size=0.1),
                                   marker=dict(color=c,
                                   line=dict(color="rgba(0, 0, 0, 1)",
                                             width=1.5)),
                                   opacity=0.7,
                                   legend='legend2'),
                      row=2, col=1)

    if not isinstance(trace_params, type(None)):
        fig.update_traces(**trace_params)

    fig.update_xaxes(mirror=True,
                     title=dict(text='Band gap, eV',
                                font=dict(size=28)),
                     row=1, col=1)
    fig.update_yaxes(mirror=True,
                     title=dict(text='Count',
                                font=dict(size=28)),
                     row=1, col=1)
    fig.update_xaxes(mirror=True,
                     title=dict(text='Band gap, eV',
                                font=dict(size=28)),
                     row=2, col=1)
    fig.update_yaxes(mirror=True,
                     title=dict(text='Count',
                                font=dict(size=28)),
                     row=2, col=1)
    
    fig.update_layout(barmode='overlay',
                      width=1200, height=1400,
                      template='simple_white',
                      font=dict(size=16),
                      legend1=dict(x=.775, y=.985,
                                   title=dict(text='X:',
                                              side='top center',
                                              font=dict(size=32)),
                                   font=dict(size=32)),
                      legend2=dict(x=.775, y=.35,
                                   title=dict(text='M:',
                                              side='top center',
                                              font=dict(size=32)),
                                   font=dict(size=32)),
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig



def make_anion_formula(metal: str,
                       halogen: str,
                       ) -> str:
    ''' ________________________________________________________________________________________________________________________________
    Function for anion composition string rendering. Use HTML tags for register.

    Parameters:
        metal (str): Composition by metal in {MX4}– anion ('Bi', 'Sb' or 'Bi Sb').
        halogen (str): Composition by halogen in {MX4}– anion ('I', 'Br', 'Cl', 'Br I' or 'Cl Br').

    Return:
        str: Anion composition string in HTML format.
    ________________________________________________________________________________________________________________________________'''
    metal = metal.replace(' ', '<sub>1–x</sub>') + '<sub>x</sub>' if len(metal) > 2 else metal
    halogen = ' '.join(sorted(halogen.split(' '), key=lambda x: ['I', 'Br', 'Cl'].index(x))).replace(' ', '<sub>4–y</sub>') + '<sub>y</sub>' if len(halogen) > 2 else halogen + '<sub>4</sub>'
    
    return metal + halogen



def descriptor_violin_plot(dataframe: pd.DataFrame,
                           traces_params: Union[Dict, None] = None,
                           layout_params: Union[Dict, None] = None,
                           ) -> go.Figure:
    '''________________________________________________________________________________________________________________________________
    Draw violin plots for all descriptors in dataset.

    Parameters:
        dataframe (pandas.DataFrame): Input dataframe.
        traces_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.
    
    Return:
        go.Figure: Violin plots for descriptors.
    ________________________________________________________________________________________________________________________________
    '''
    fig = make_subplots(rows=1, cols=6,
                        column_widths=[1/30, 10/30, 12/30, 3/30, 2/30, 2/30],
                        horizontal_spacing=0.03)
    
    fig.add_trace(go.Violin(y=dataframe['Temperature'],
                            name='Temperature',
                            showlegend=False,
                            width=.75,
                            box_visible=True,
                            meanline_visible=True,
                            ),
                  row=1, col=1)

    for col in dataframe.columns[2:8].to_list()+['d_average']+dataframe.columns[-4:-1].to_list():
        fig.add_trace(go.Violin(y=dataframe[col],
                                name=visualization_descriptor_names[col],
                                showlegend=False,
                                width=.75,
                                box_visible=True,
                                meanline_visible=True,
                                ),
                      row=1, col=2)

    for col in dataframe.columns[8:11].to_list()+dataframe.columns[12:13].to_list()+dataframe.columns[14:21].to_list()+dataframe.columns[22:23].to_list():
        fig.add_trace(go.Violin(y=dataframe[col],
                                name=visualization_descriptor_names[col],
                                showlegend=False,
                                width=0.75,
                                box_visible=True,
                                meanline_visible=True,
                                ),
                      row=1, col=3)
    
    for col in dataframe.columns[[11, 13, 21]]:
        fig.add_trace(go.Violin(y=dataframe[col],
                                name=visualization_descriptor_names[col],
                                showlegend=False,
                                width=0.75,
                                box_visible=True,
                                meanline_visible=True,
                                ),
                      row=1, col=4)
    
    fig.add_trace(go.Violin(y=dataframe['delta_d']*10000, name='∆d, 10<sup>4</sup>',
                            showlegend=False,
                            width=0.75,
                            box_visible=True,
                            meanline_visible=True,
                            ),
                  row=1, col=5)
    fig.add_trace(go.Violin(y=dataframe['sigma_2'], name='σ<sup>2</sup>',
                            showlegend=False,
                            width=0.75,
                            box_visible=True,
                            meanline_visible=True,
                            ),
                  row=1, col=5)
    
    for col in['N_XX', 'N_VdW']:
        fig.add_trace(go.Violin(y=dataframe[col],
                                name=visualization_descriptor_names[col],
                                showlegend=False,
                                width=0.75,
                                box_visible=True,
                                meanline_visible=True,
                                ),
                      row=1, col=6)
    
    if not isinstance(traces_params, type(None)):
         fig.update_trace(**traces_params)

    fig.update_xaxes(tickangle=45,
                     tickfont=dict(size=16),
                     )
    fig.update_yaxes(title='Temperatures, K',
                     title_standoff=5,
                     row=1, col=1)
    fig.update_yaxes(title='Distances, Å',
                     title_standoff=5,
                     row=1, col=2)
    fig.update_yaxes(title='Angles, °',
                     title_standoff=5,
                     row=1, col=3)
    fig.update_yaxes(title='Angles, °',
                     title_standoff=5,
                     row=1, col=4)
    fig.update_layout(width=1700, height=700,
                      template='plotly_white',
                      colorway=['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 30)],
                      title=dict(text='Distribution of descriptors',
                                 x=.5,
                                 font=dict(size=28),
                                 ))

    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig



def heatmap_plot(corr_map: pd.DataFrame,
                 traces_params: Union[Dict, None] = None,
                 layout_params: Union[Dict, None] = None,
                 ) -> go.Figure:
    '''________________________________________________________________________________________________________________________________
    Build Heatmap for input descriptor matrix.
    
    Parameters:
        corr_map (pandas.DataFrame): Input descriptor matrix.
        traces_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.


    Return:
        go.Figure: Heatmap as plotly graph object. 
    ________________________________________________________________________________________________________________________________'''
    corr_map = corr_map.rename(index=visualization_descriptor_names, columns=visualization_descriptor_names)

    fig = go.Figure(go.Heatmap(x=corr_map.columns[:-1],
                               y=corr_map.index[1:],
                               z=corr_map[1:],
                               zmin=-1,
                               zmax=1,
                               zmid=0,
                               texttemplate='%{z:.4f}',
                               colorscale='RdBu_r',
                               ))
    
    if not isinstance(traces_params, type(None)):
        fig.update_traces(**traces_params)

    fig.update_layout(width=1500, height=1500,
                      template = 'plotly_white',
                      xaxis=dict(side='top',
                                 showgrid=False),
                      yaxis=dict(showgrid=False),
                      title=dict(text='Heatmap',
                                 x=0.5,
                                 font=dict(size=28),
                                 ))
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig



def descriptor_pairwise(dataframe: pd.DataFrame,
                        composition_aid: pd.DataFrame,
                        feature_names: Union[ArrayLike, Sequence],
                        traces_params: Union[Dict, None] = None,
                        layout_params: Union[Dict, None] = None,
                        ) -> go.Figure:
    ''''________________________________________________________________________________________________________________________________
    Build pairwise plots for selected features.
    
    Parameters:
        dataframe (pandas.DataFrame): Input feature matrix.
        composition_aid (pandas.DataFrame): Dataframe for composition rendering.
        feature_names (ArrayLike | Sequence): Array with name of features.
        traces_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.


    Return:
        go.Figure: Pairwise plots. 
    ________________________________________________________________________________________________________________________________'''
    fig = go.Figure(go.Splom(dimensions=[dict(label=visualization_descriptor_names[feature], values=dataframe[feature]) for feature in feature_names],
                             showupperhalf=False,
                             diagonal_visible=False,
                             text=pd.concat([dataframe[['Refcode', 'Temperature']], composition_aid], axis=1).apply(lambda row: row['Refcode']+'<br>'+make_anion_formula(row['M'],row['X'])+'<br>'+str(row['Temperature'])+' K', axis=1),
                             marker=dict(color=pd.factorize(pd.Series([x[0]+x[1] for x in zip(composition_aid.M, composition_aid.X)]))[0],
                                         showscale=False,
                                         line_color='white',
                                         line_width=0.5,
                                         ),
                             ))
    
    if not isinstance(traces_params, type(None)):
         fig.update_traces(**traces_params)

    fig.update_xaxes(mirror=True,
                     showgrid=True,
                     )
    fig.update_yaxes(mirror=True,
                     showgrid=True,
                     )
    fig.update_layout({'xaxis'+str(i+1): {'mirror':"all", 'showgrid': True} for i in range(16)})
    fig.update_layout({'yaxis'+str(i+1): {'mirror':"all", 'showgrid': True} for i in range(16)})
    fig.update_layout(width=1500, height=1500,
                      template='simple_white',
                      title=dict(text='Pairwise for selected descriptors',
                                 x=.5,
                                 font=dict(size=24),
                                 ))
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig



def mx_correlation_plot(data: pd.DataFrame,
                        error_str: ArrayLike,
                        color_list: ArrayLike = ["#053061", "#C6CFFB", "#0C6A5E","#C8F1D0","#B2182B","#FFA894","#053061", "#7A85BA","#0C6A5E","#79AB83","#B2182B","#D47661"],
                        layout_params: Union[Dict, None] = None,
                        traces_params: Union[Dict, None] = None,
                        ) -> go.Figure:
    '''______________________________________________________________________________________________________________________________
    Draw plot that demonstrate correlations for M–X distances in trans-X–M–X fragment for each anion composition and calculate
    Spearman correlation coefficients.

    arameters:
        data (pd.DataFrame): Matrix that contain M–X distances, composition vector and Refcode.
        error_str (ArrayLike): Array with refcodes problematic structure. 
        color_list(ArrayLike): Array of 12 colors; 6 for traces and 6 for title.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating, if None – default style.
        traces_params (Dict | None, default=None): Dictinary with parameters for trace updating, if None – default style.


    Return:
        plotly.graph_objects.Figure: Pairwise M–X bond-length relationships plot as one plotly.graph_objects.Figure object.
    '''
    fig = go.Figure()

    count = 0

    for x_composition in ['I', 'Br', 'Cl']:
        for m_composition in ['Bi', 'Sb']:
            mx_name_lens = data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].shape[0]
            mx1 = pd.concat([data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['X_t1'],
                             data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['X_t2'],
                             data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['X_d2'],
                             ])
            mx2 = pd.concat([data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['X_d3'],
                             data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['X_d1'],
                             data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['X_d4'],
                             ])
            fig.add_trace(go.Scatter(x=mx1,
                                     y=mx2,
                                     text=pd.concat([data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['Refcode'],
                                                     data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['Refcode'],
                                                     data.iloc[data[(data['X']==x_composition)&(data['M']==m_composition)&(data['MX4 type']=='alpha')&(~data['Refcode'].isin(error_str))].index]['Refcode'],
                                                     ]),
                                     mode='markers',
                                     name=f'{m_composition}{x_composition}<sub>4</sub>',
                                     hovertemplate=pd.Series([visualization_descriptor_names['X_t1']]*mx_name_lens+[visualization_descriptor_names['X_t2']]*mx_name_lens+[visualization_descriptor_names['X_d2']]*mx_name_lens)+': %{x:.4f} Å<br>'+pd.Series([visualization_descriptor_names['X_d3']]*mx_name_lens+[visualization_descriptor_names['X_d1']]*mx_name_lens+[visualization_descriptor_names['X_d4']]*mx_name_lens)+': %{y:.4f} Å<br>%{text}',
                                     marker=dict(size=11,
                                                 color=color_list[count],
                                                 line_color="#000000",
                                                 line_width=1.5,
                                                 )
                                            
                                    ))
            fig.add_annotation(go.layout.Annotation({'text': f"<i>ρ<sub>S</sub></i> = {str(round(mx1.corr(mx2, method='spearman'), 4)).ljust(7, '0')}",
                                                     'x': 3.335 - count*0.04,
                                                     'y': mx2.min() + [0.035, -0.025, 0.03, -0.04, -0.085, -0.01][count],
                                                     'align': 'left', 'showarrow': False,
                                                     'font': {'size': 28, "color": color_list[count+6]}
                                                     }))
            count += 1
    
    if not isinstance(traces_params, type(None)):
         fig.update_traces(**traces_params)

    fig.update_xaxes(range=[2.3, 3.5],
                     mirror=True,
                     dtick=.1,
                     tickformat='.1f',
                     title_standoff=20,
                     title=dict(text='d(M–X), Å',
                                font=dict(size=28))
                     )
    fig.update_yaxes(range=[2.3, 3.9],
                     mirror=True,
                     title_standoff=20,
                     title=dict(text='d(M–X), Å',
                                font=dict(size=28))
                     )
    fig.update_layout(width=1400, height=800,
                      template='simple_white',
                      margin=dict(t=50, l=120),
                      font=dict(size=24),
                      legend=dict(x=0.88, y=1, font=dict(size=32))
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)
        
    return fig



def descriptors_vs_bandgap(dataframe: pd.DataFrame,
                           feature_1: str,
                           feature_2: str,
                           composition_aid: pd.DataFrame,
                           subplot_titles: Union[ArrayLike, Sequence, None] = None,
                           colorscale: str = 'RdBu_r',
                           traces_params: Union[Dict, None] = None,
                           layout_params: Union[Dict, None] = None,
                           ) -> go.Figure:
    '''________________________________________________________________________________________________________________________________
    Build plots Descriptor vs Band gap for two descriptors.
    
    Parameters:
        dataframe (pandas.DataFrame): Input feature matrix.
        feature_1 (str): Feature name for axis 1.
        feature_2 (str): Feature name for axis 2.
        composition_aid (pandas.DataFrame): Dataframe for composition rendering.
        colorscale (str, default="RdBu_r"): Colour scheme for plots, may be any plotly color scale (see,
                                            https://plotly.com/python/builtin-colorscales/).
        traces_params (Dict | None, default=None): Dictinary with parameters for heatmap trace updating or None.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating or None.


    Return:
        go.Figure: Plots Descriptor vs Band gap. 
    ________________________________________________________________________________________________________________________________
    '''
    if not isinstance(subplot_titles, type(None)):
        fig = make_subplots(rows=1, cols=2,
                            horizontal_spacing=0.05,
                            vertical_spacing=0.15,
                            subplot_titles=subplot_titles
                            )
    else:
        fig = make_subplots(rows=1, cols=2,
                            horizontal_spacing=0.05,
                            )
        
    fig.update_annotations(font=dict(size=28))

    fig.add_trace(go.Scatter(x=dataframe[feature_1], y=dataframe['Bandgap'],
                             mode='markers',
                             text=pd.concat([dataframe[['Refcode', 'Temperature']], composition_aid], axis=1).apply(lambda row: row['Refcode']+'<br>'+make_anion_formula(row['M'],row['X'])+'<br>'+str(row['Temperature']), axis=1),
                             showlegend=False,
                             name=f'{visualization_descriptor_names[feature_1]} vs Band gap',
                             hovertemplate = visualization_descriptor_names[feature_1]+': %{x:.5f}<br>Band gap: %{y:.2f}<br>%{text}',
                             marker=dict(size=12.5,
                                         color=pd.factorize(pd.Series([x[0]+x[1] for x in zip(composition_aid.M, composition_aid.X)]))[0],
                                         colorscale=colorscale,
                                         line_color="#000000",
                                         line_width=1.5,)
                             ),
                    row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe[feature_2], y=dataframe['Bandgap'],
                             mode='markers',
                             text=pd.concat([dataframe[['Refcode', 'Temperature']], composition_aid], axis=1).apply(lambda row: row['Refcode']+'<br>'+make_anion_formula(row['M'],row['X'])+'<br>'+str(row['Temperature']), axis=1),
                             showlegend=False,
                             name=f'{visualization_descriptor_names[feature_2]} vs Band gap',
                             hovertemplate = visualization_descriptor_names[feature_2]+': %{x:.3f}<br>Band gap: %{y:.2f}<br>%{text}',
                             marker=dict(size=12.5,
                                         color=pd.factorize(pd.Series([x[0]+x[1] for x in zip(composition_aid.M, composition_aid.X)]))[0],
                                         colorscale=colorscale,
                                         line_color="#000000",
                                         line_width=1.5,
                                         )),
                    row=1, col=2)

    # Legend
    fig.add_trace(go.Scatter(x=[None], y=[None], name='BiI<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [0]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='SbI<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [1/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='Bi<sub>1–x</sub>Sb<sub>x</sub>I<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [2/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='BiI<sub>4–y</sub>Br<sub>y</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [3/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='SbI<sub>4–y</sub>Br<sub>y</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [4/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='Bi<sub>1–x</sub>Sb<sub>x</sub>I<sub>4–y</sub>Br<sub>y</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [5/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='BiBr<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [6/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='SbBr<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [7/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='BiCl<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [8/9]), line_color="#000000", line_width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='SbCl<sub>4</sub>', mode='markers', marker=dict(size=20, color=px.colors.sample_colorscale(colorscale, [1]), line_color="#000000", line_width=2)))

    if not isinstance(traces_params, type(None)):
         fig.update_traces(**traces_params)

    fig.update_xaxes(row=1, col=1,
                     range=[0, 0.025],
                     mirror=True,
                     showgrid=True,
                     title=dict(text='∆d',
                                font=dict(size=28))
                     )
    fig.update_xaxes(row=1, col=2,
                     range=[0.9, 1.3],
                     mirror=True,
                     showgrid=True,
                     dtick=.1,
                     tickformat='.1f',
                     title=dict(text='d<sub>average</sub>(X···X)/2·r<sub>VdW</sub>(X)',
                                font=dict(size=28))
                     )

    fig.update_yaxes(row=1, col=1,
                     range=[1.6, 3.6],
                     mirror=True,
                     showgrid=True,
                     dtick=.4,
                     tickformat='.1f',
                     title=dict(text='Band gap, eV',
                                font=dict(size=28))
                     )
    fig.update_yaxes(row=1, col=2,
                     range=[1.6, 3.6],
                     mirror=True,
                     showgrid=True,
                     dtick=.4,
                     ticklen=0,
                     showticklabels=False
                     )
    fig.update_layout(width=1400, height=750,
                      template='simple_white',
                      margin=dict(t=100),
                      font=dict(size=18),
                      legend=dict(x=0.32, y=0,
                                  font=dict(size=20))
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)
    
    return fig

########################################################################################################################################



####################################################### CODE FOR MACHINE LERNING #######################################################

def loo_oof_predictions(pipe: Pipeline,
                        X: pd.DataFrame,
                        y: Union[pd.Series, np.ndarray],
                        ) -> np.ndarray:
    '''________________________________________________________________________________________________________________________________
    Generate out-of-fold (OOF) predictions using Leave-One-Out cross-validation.

    Parameters:
          pipe (sklearn.pipeline.Pipeline): Pipeline including preprocessing and regression model.
          X (pandas.DataFrame): Descriptor matrix.
          y (pandas.Series | numpy.ndarray): Target vector.

    Returns:
          numpy.ndarray: Out-of-fold predictions of shape (n_samples,).
    ________________________________________________________________________________________________________________________________'''
    loo = LeaveOneOut()
    y_oof = np.zeros(len(y))

    # # For PLSRegression (Uncomment next line):
    # y_used = y_train.values.reshape(-1,1) if model_name == 'PLS' else y_train

    for train_idx, val_idx in loo.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        if isinstance(y, np.ndarray):
            y_tr = y[train_idx]
        else:
            y_tr = y.iloc[train_idx]

        pipe_clone = clone(pipe)
        pipe_clone.fit(X_tr, y_tr)
        y_pred_val = pipe_clone.predict(X_val)
        y_oof[val_idx] = y_pred_val #.ravel() # Uncomment for PLCRegression using:

    return y_oof



def objective(trial: optuna.trial.Trial,
              model_name: str,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              score_function: Callable[[ArrayLike, ArrayLike], float] = root_mean_squared_error,
              seed: int = 7,
              ) -> float:
    '''________________________________________________________________________________________________________________________________
    Optuna objective function for hyperparameter optimization based on LOO-CV with out-of-fold (OOF) predictions.
    
    Parameters:
          trial (optuna.trial.Trial): Optuna trial object used to sample hyperparameters.
          model_name (str): Name of the model to optimize (e.g. 'LR', 'SVR', 'GPR', 'XGBoost').
          X_train (pd.DataFrame): Training feature matrix.
          y_train (pd.Series): Training target vector.
          score_function (Callable[[ArrayLike, ArrayLike] -> float, default=sklearn.metrics.root_mean_squared_error): Metric function with signature score(y_true, y_pred).
          seed (int, default=7): Random seed for reproducibility.

    Returns:
          float: Cross-validated score computed from LOO out-of-fold predictions. Lower is better – Optuna direction="minimize".
    ________________________________________________________________________________________________________________________________'''
    if model_name == 'Linear':
        model = LinearRegression()
    elif model_name == 'GLR':
        power = trial.suggest_float('power', 0.0, 2.0)
        alpha = trial.suggest_float('alpha', 1e-6, 10.0, log=True)
        model = TweedieRegressor(power=power, alpha=alpha, max_iter=1000)
    elif model_name == 'ElasticNet':
        alpha = trial.suggest_float('alpha', 1e-5, 10.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    elif model_name == 'Huber':
        epsilon = trial.suggest_float('epsilon', 1.1, 2.0)
        alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
        model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)
    elif model_name == 'BayesianRidge':
        alpha_1 = trial.suggest_float('alpha_1', 1e-6, 1e-2, log=True)
        alpha_2 = trial.suggest_float('alpha_2', 1e-6, 1e-2, log=True)
        lambda_1 = trial.suggest_float('lambda_1', 1e-6, 1e-2, log=True)
        lambda_2 = trial.suggest_float('lambda_2', 1e-6, 1e-2, log=True)
        model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, fit_intercept=True)
     # # PLSRegression (Uncomment next lines):
     # elif model_name == 'PLS':
     #    n_components = trial.suggest_int('n_components', 2, min(10, X_train1.shape[1]))
     #    model = PLSRegression(n_components=n_components)
    elif model_name == 'SVR':
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        model = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree, gamma=gamma)
    elif model_name == 'GPR':
        kernel_choice = trial.suggest_categorical('kernel_choice', ['RBF', 'Matern', 'RationalQuadratic', 'DotProduct'])
        if kernel_choice == 'RBF':
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
            kernel = RBF(length_scale=length_scale)
        elif kernel_choice == 'Matern':
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
            nu = trial.suggest_float('nu', 0.5, 2.5)
            kernel = Matern(length_scale=length_scale, nu=nu)
        elif kernel_choice == 'RationalQuadratic':
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
            alpha_rq = trial.suggest_float('alpha', 0.1, 10.0, log=True)
            kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha_rq)
        elif kernel_choice == 'DotProduct':
            sigma_0 = trial.suggest_float('sigma_0', 0.1, 5.0)
            kernel = DotProduct(sigma_0=sigma_0)
        alpha = trial.suggest_float('alpha', 1e-7, 1.0, log=True)
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, optimizer=None, random_state=seed)
    elif model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 15)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1)
    elif model_name == 'ExtraTrees':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        max_features = trial.suggest_float('max_features', 0.5, 1.0)
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=seed,
                                    n_jobs=-1)
    elif model_name == 'DecisionTree':
        max_depth = trial.suggest_int('max_depth', 2, 6)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=seed)
    elif model_name == 'XGBoost':
        n_estimators=100
        max_depth = trial.suggest_int('max_depth', 2, 5)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 0.9)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 0.9)
        min_child_weight = trial.suggest_int('min_child_weight', 3, 10)
        reg_alpha=1.0
        reg_lambda=5.0
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, 
                                 subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight, 
                                 reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                 random_state=seed, eval_metric='rmse')
    else:
        print(f'Unknown model – {model_name}')
        return None
     
    try:
        # Pipeline
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

        # LOO-CV
        y_oof = loo_oof_predictions(pipe, X_train, y_train)
        
        # Calculate out-of-fold (OOF) metric
        scores = score_function(y_train, y_oof)

        return scores
    
    except Exception as e:
        print(f'Trial failed for {model_name}: {e}')

        return None



def hyperparameter_optimization(models_dict: Dict[str, Callable],
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                score_function: Callable[[ArrayLike, ArrayLike], float] = root_mean_squared_error,
                                seed: int = 7,
                                n_trials: int = 100,
                                ) -> Dict[str, Dict]:
    '''________________________________________________________________________________________________________________________________
    Run Optuna hyperparameter optimization for multiple models.

    Parameters:
          models_dict (Dict): Dictionary mapping model names to model constructors.
          X_train (pandas.DataFrame): Training feature matrix.
          y_train (pandas.Series): Training target vector.
          score_function (Callable[[ArrayLike, ArrayLike] -> float, default=sklearn.metrics.root_mean_squared_error): Metric function with signature score(y_true, y_pred).
          seed (int, default=7): Random seed for reproducibility.
          n_trials (int, default=50): Number of Optuna trials per model.

    Returns:
          Dict: Dictionary mapping model names to best hyperparameters.
    ________________________________________________________________________________________________________________________________'''
    best_trials: Dict[str, Dict] = {}

    for model_name in models_dict.keys():
        print(f'Optimizing {model_name}...')

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))

        # Create Optuna objective function for a specific model:
        objective_fn = lambda trial:  objective(trial, model_name=model_name, X_train=X_train, y_train=y_train, score_function=score_function, seed=seed)

        study.optimize(objective_fn, n_trials=n_trials, catch=(Exception,))
        best_trials[model_name] = study.best_params

        print(f'Best params for {model_name}:', f'{study.best_params},', f'Score: {study.best_value:.4f}')
        print()

    return best_trials



def build_final_pipelines(hyperparameters_dict: Dict[str, Dict],
                          scaler: Callable = StandardScaler,
                          seed: int = 7,
                          ) -> Dict[str, Pipeline]:
    """_______________________________________________________________________________________________________________________________
    Build final pipelines with best hyperparameters for each model.

    Parameters:
        hyperparameters_dict (Dict[str, Dict]): Dictionary with hyperparameters for each studied model.
        scaler (Callable, default=sklearn.preprocessing.StandardScaler): Skikit-learn object for descriptor standardization.
        seed (int, default=7): Random seed for reproducibility.
    
    Returns:
        Dict[str, sklearn.pipeline.Pipeline]: Dictionary of final pipelines for each model.
    _______________________________________________________________________________________________________________________________"""

    final_pipelines: Dict[str, Pipeline] = {}

    for model_name, params in hyperparameters_dict.items():
        if model_name == 'Linear':
            model = LinearRegression()
        elif model_name == 'GLR':
            model = TweedieRegressor(**params, max_iter=1000)
        elif model_name == 'ElasticNet':
            model = ElasticNet(**params, max_iter=10000)
        elif model_name == 'Huber':
            model = HuberRegressor(**params, max_iter=1000)
        elif model_name == 'BayesianRidge':
            model = BayesianRidge(**params, fit_intercept=True)
        # # PLSRegression (Uncomment next lines):
        # elif model_name == 'PLS':
        #     model = PLSRegression(**params)
        elif model_name == 'SVR':
            model = SVR(**params)
        elif model_name == 'GPR':
            kernel = {'RBF': RBF, 'Matern': Matern, 'RationalQuadratic': RationalQuadratic, 'DotProduct': DotProduct}[hyperparameters_dict[model_name]['kernel_choice']]
            kernel = kernel(**{k: hyperparameters_dict[model_name][k] for k in list(filter(lambda x: x not in ['kernel_choice', 'alpha'], hyperparameters_dict[model_name].keys()))})
            model = GaussianProcessRegressor(kernel=kernel, alpha=hyperparameters_dict[model_name]['alpha'], optimizer=None, random_state=seed)
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(**params, random_state=seed, n_jobs=-1)
        elif model_name == 'ExtraTrees':
            model = ExtraTreesRegressor(**params, random_state=seed, n_jobs=-1)
        elif model_name == 'DecisionTree':
            model = DecisionTreeRegressor(**params, random_state=seed)
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(**params, n_estimators=100, reg_alpha=1.0, reg_lambda=5.0, random_state=seed, eval_metric='rmse')
        
        pipe = Pipeline([('scaler', scaler()), ('model', model)])
        final_pipelines[model_name] = pipe

    return final_pipelines



def train_validate_models(final_pipelines: Dict[str, Pipeline],
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          ) -> Tuple[Dict]:
    '''_______________________________________________________________________________________________________________________________
    Train and LOO-CV for each final model.
    
    Parameters:
        final_pipelines (Dict[str, Pipeline]): Dictionary of final pipelines for each model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        
    Return:
        Dict: Dictionaries with predidicted values on train subsets.
        Dict: Dictionaries with metrics on training and validation.
    _______________________________________________________________________________________________________________________________'''
    predicted_values = {'train': {}, 'test': {}}
    results = {}
    for model_name, pipe in final_pipelines.items():
        # LOO-CV:
        print(f'LOO-CV for {model_name}')
        y_oof = loo_oof_predictions(pipe=pipe, X=X_train, y=y_train)
        rmse_val = root_mean_squared_error(y_train, y_oof)
        print(f'RMSE on LOO-CV: {round(rmse_val, 4)}')
        print()

        # Final model training
        print(f'Train final {model_name}')
        print()
        pipe.fit(X_train, y_train) #
        # # For PLSRegression (Uncomment next line):
        # y_used = y_train.values.reshape(-1,1) if model_name == 'PLS' else y_train
        # pipe.fit(X_train, y_used)

        y_train_pred = pipe.predict(X_train) #.ravel() # Uncomment for PLCRegression using:
        predicted_values['train'][model_name] = y_train_pred
        results[model_name] = {'MAE on train': mean_absolute_error(y_train, y_train_pred),
                               'RMSE on train': root_mean_squared_error(y_train, y_train_pred),
                               'R2 on train': r2_score(y_train, y_train_pred),
                               'MAE on validation': mean_absolute_error(y_train, y_oof),
                               'RMSE on validation': rmse_val,
                               'R2 on validation': r2_score(y_train, y_oof),
                               }
    
    return predicted_values, results



def train_and_validate(hyperparameters_dict: Dict[str, Dict],
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       scaler: Callable = StandardScaler,
                       seed: int = 7,) -> Tuple[Dict]:
    '''_______________________________________________________________________________________________________________________________
    Train and validete models (full pipeline).
    
    Parameters:
        hyperparameters_dict (Dict[str, Dict]): Dictionary with hyperparameters for each studied model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        scaler (Callable, default=sklearn.preprocessing.StandardScaler): Skikit-learn object for descriptor standardization.
        seed (int, default=7): Random seed for reproducibility.
        
    Return:
        Dict: Dictionaries with final pipelines of trained models.
        Dict: Dictionaries with predidicted values on train subset,
        Dict: Dictionaries with metrics on training and validation.
    _______________________________________________________________________________________________________________________________'''
    final_pipelines = build_final_pipelines(hyperparameters_dict=hyperparameters_dict, scaler=scaler, seed=seed)
    predicted_values, results = train_validate_models(final_pipelines=final_pipelines, X_train=X_train, y_train=y_train)

    return final_pipelines, predicted_values, results



def predict_models(final_pipelines: Dict[str, Pipeline],
                   predicted_values: Dict[str, ArrayLike], 
                   results: Dict[str, ArrayLike],
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   ) -> Tuple[Dict]:
    '''_______________________________________________________________________________________________________________________________
    Make prediction on test subset for each model.
    
    Parameters:
        final_pipelines (Dict[str, sklearn.pipeline.Pipeline]): Dictinary with final pipelines of trained models.
        predicted_values (Dict[str, ArrayLike]): Dictionary with predidicted values on train subset.
        results (Dict[str, ArrayLike]): Dictioary with with metrics on training and validation.
        X_train (pandas.DataFrame): Training feature matrix.
        X_test (pandas.DataFrame): Testing feature matrix.
        y_test (pandas.DataFrame): Testing target vector.

    Reyurn:
        Dict: Dictionary with predidicted values on train and test subsets.
        Dict: Dictionary with final metrics.
    _______________________________________________________________________________________________________________________________'''
    for model_name, pipe in final_pipelines.items():
        y_pred = pipe.predict(X_test[X_train.columns])
        predicted_values['test'][model_name] = y_pred
        rmse_test = root_mean_squared_error(y_test, y_pred)
        results[model_name] = results[model_name] | {'MAE on test': mean_absolute_error(y_test, y_pred),
                                                     'RMSE on test': rmse_test,
                                                     'R2 on test': r2_score(y_test, y_pred),
                                                     }
        print(f'RMSE on test for {model_name}: {round(rmse_test, 4)}')
        print()

    return predicted_values, results



def single_predicted_vs_actual_plot(y_train_pred: ArrayLike,
                                    y_train_true: ArrayLike,
                                    y_test_pred: ArrayLike,
                                    y_test_true: ArrayLike,
                                    train_text: Union[ArrayLike, Sequence],
                                    test_text: Union[ArrayLike, Sequence],
                                    layout_params: Union[Dict, None] = None,
                                    identity_line_params: Union[Dict, None] = None,
                                    train_trace_params: Union[Dict, None] = None,
                                    test_trace_params: Union[Dict, None] = None,
                                    ) -> go.Figure:
    '''______________________________________________________________________________________________________________________________
    Draw Predicted vs Actual plot for one model.
    
    Parameters:
        y_train_pred (ArrayLike): Vector of predicted target values on train subset.
        y_train_true (ArrayLike): Vector of actual target values on train subset.
        y_test_pred (ArrayLike): Vector of predicted target values on test subset.
        y_test_true (ArrayLike): Vector of actual target values on test subset.
        train_text (ArrayLike | Sequence): Array with text for train points.
        test_text (ArrayLike | Sequence): Array with text for text points.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating, if None – default style.
        identity_line_params (Dict | None, default=None): Dictinary with parameters for identity line updating, if None – default style.
        train_trace_params (Dict | None, default=None): Dictinary with parameters for train trace updating, if None – default style.
        test_trace_params (Dict | None, default=None): Dictinary with parameters for test trace updating, if None – default style.


    Return:
        plotly.graph_objects.Figure: Predicted vs Actual plot as plotly.graph_objects.Figure object.
    ______________________________________________________________________________________________________________________________'''
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(0, 4, 0.01),
                             y=np.arange(0, 4, 0.01),
                             showlegend=False,
                             name='Identity line',
                             hovertemplate='',
                             mode='lines',
                             line=dict(width=3,
                                       dash='dash',
                                       color='black',
                                       ),
                             ))
    fig.add_trace(go.Scatter(x=y_train_true,
                             y=y_train_pred,
                             name='train',
                             text=train_text,
                             hovertemplate='train<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                             mode='markers',
                             marker=dict(size=12.5,
                                         symbol='square',
                                         color='#4C78A8',
                                         line_width=2.5,
                                         line_color='#0B1532',
                                         ),
                             ))
    fig.add_trace(go.Scatter(x=y_test_true,
                             y=y_test_pred,
                             name='test',
                             text=test_text,
                             hovertemplate='test<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                             mode='markers',
                             marker=dict(size=17.5,
                                         symbol='star',
                                         color='#E33333',
                                         line_width=2.5,
                                         line_color='#550B0B',
                                         ),
                             ))
    
    if not isinstance(identity_line_params, type(None)):
        fig.data[0].update(**identity_line_params)
    if not isinstance(train_trace_params, type(None)):
        fig.data[1].update(**train_trace_params)
    if not isinstance(test_trace_params, type(None)):
        fig.data[2].update(**test_trace_params)

    fig.update_xaxes(range=[1.6, 3.6],
                     mirror=True,
                     dtick=.2,
                     tickformat='.1f',
                     title=dict(text='Band gap experimental, eV',
                                font=dict(size=32),
                                ))
    fig.update_yaxes(range=[1.6, 3.6],
                     mirror=True,
                     dtick=.2,
                     tickformat='.1f',
                     title=dict(text='Band gap predicted, eV',
                                font=dict(size=32),
                                ))
    fig.update_layout(width=1200, height=1000,
                      template='simple_white',
                      font_size=20,
                      legend=dict(x=.05, y=1,
                                  font=dict(size=32),
                                  ),
                      margin=dict(t=80),
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig



def double_predicted_vs_actual_plot(y_train_pred_1: ArrayLike,
                                    y_train_pred_2: ArrayLike,
                                    y_train_true: ArrayLike,
                                    y_test_pred_1: ArrayLike,
                                    y_test_pred_2: ArrayLike,
                                    y_test_true: ArrayLike,
                                    results_1: pd.Series,
                                    results_2: pd.Series,
                                    train_text: Union[ArrayLike, Sequence],
                                    test_text: Union[ArrayLike, Sequence],
                                    subplot_titles: Union[ArrayLike, Sequence],
                                    layout_params: Union[Dict, None] = None,
                                    traces_params: Union[Dict, None] = None,
                                    ) -> go.Figure:
    '''______________________________________________________________________________________________________________________________
    Draw Predicted vs Actual plots for two models.
    
    Parameters:
        y_train_pred_1 (ArrayLike): Vector of predicted target values on train subset for axis 1.
        y_train_pred_2 (ArrayLike): Vector of predicted target values on train subset for axis 2.
        y_train_true (ArrayLike): Vector of actual target values on train subset.
        y_test_pred_1 (ArrayLike): Vector of predicted target values on test subset for axis 1.
        y_test_pred_2 (ArrayLike): Vector of predicted target values on test subset for axis 2.
        y_test_true (ArrayLike): Vector of actual target values on test subset.
        results_1 (pandas.Series): Series with metrics on training and validation for axis 1.
        results_2 (pandas.Series): Series with metrics on training and validation for axis 2.
        train_text (ArrayLike | Sequence): Array with text for train points.
        test_text (ArrayLike | Sequence): Array with text for text points.
        subplot_titles (ArrayLike | Sequence): Array with titles for subplots.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating, if None – default style.
        traces_params (Dict | None, default=None): Dictinary with parameters for trace updating, if None – default style.


    Return:
        plotly.graph_objects.Figure: Two Predicted vs Actual plots as one plotly.graph_objects.Figure object.
    ______________________________________________________________________________________________________________________________'''
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=subplot_titles)
    fig.update_annotations(font=dict(size=28))

    # 1
    fig.add_trace(go.Scatter(x=np.arange(0, 4, 0.01),
                             y=np.arange(0, 4, 0.01),
                             name='Identity line',
                             showlegend=False,
                             hovertemplate='',
                             mode='lines',
                             line=dict(width=3,
                                       dash='dash',
                                       color='black',
                                       ),
                             ),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=y_train_true,
                             y=y_train_pred_1,
                             legend='legend1',
                             name='train',
                             text=train_text,
                             hovertemplate='train<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                             mode='markers',
                             marker=dict(size=12.5,
                                         symbol='square',
                                         color='#4C78A8',
                                         line_width=2.5,
                                         line_color='#0B1532',
                                         ),
                             ),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test_true,
                             y=y_test_pred_1,
                             legend='legend1',
                             name='test',
                             text=test_text,
                             hovertemplate='test<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                             mode='markers',
                             marker=dict(size=17.5,
                                         symbol='star',
                                         color='#E33333',
                                         line_width=2.5,
                                         line_color='#550B0B',
                                         )
                             ),
                  row=1, col=1)
    fig.add_annotation(go.layout.Annotation({'text': f"On test:<br>MAE = {round(results_1.loc['MAE on test'], 2):.2f}<br>RMSE = {round(results_1.loc['RMSE on test'], 2):.2f}<br>R<sup>2</sup> = {round(results_1.loc['R2 on test'], 4):.4f}",
                                             'xref': 'x1',
                                             'yref': 'y1',
                                             'x': 1.95, 'y': 3.35,
                                             'align': 'left',
                                             'showarrow': False,
                                             'font': {'size': 24}
                                             }))
    
    # 2
    fig.add_trace(go.Scatter(x=np.arange(0, 4, 0.01),
                             y=np.arange(0, 4, 0.01),
                             name='Identity line',
                             showlegend=False,
                             hovertemplate='',
                             mode='lines',
                             line=dict(width=3,
                                       dash='dash',
                                       color='black',
                                       )
                             ),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=y_train_true,
                             y=y_train_pred_2,
                             legend='legend2',
                             name='train',
                             text=train_text,
                             hovertemplate='train<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                             mode='markers',
                             marker=dict(size=12.5,
                                         symbol='square',
                                         color='#4C78A8',
                                         line_width=2.5,
                                         line_color='#0B1532',
                                         )
                             ),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=y_test_true,
                             y=y_test_pred_2,
                             legend='legend2',
                             name='test',
                             text=test_text,
                             hovertemplate='text<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                             mode='markers',
                             marker=dict(size=17.5,
                                         symbol='star',
                                         color='#E33333',
                                         line_width=2.5,
                                         line_color='#550B0B',
                                         )
                             ),
                  row=1, col=2)
    fig.add_annotation(go.layout.Annotation({'text': f"On test:<br>MAE = {round(results_2.loc['MAE on test'], 2):.2f}<br>RMSE = {round(results_2.loc['RMSE on test'], 2):.2f}<br>R<sup>2</sup> = {round(results_2.loc['R2 on test'], 4):.4f}",
                                             'xref': 'x2',
                                             'yref': 'y2',
                                             'x': 1.95, 'y': 3.35,
                                             'align': 'left',
                                             'showarrow': False,
                                             'font': {'size': 24}
                                             }))

    if not isinstance(traces_params, type(None)):
        fig.update_traces(**traces_params)

    fig.update_xaxes(range=[1.6, 3.6],
                     mirror=True,
                     dtick=.2,
                     tickformat='.1f',
                     title=dict(text='Band gap experimental, eV',
                                font=dict(size=28),
                                ))
    fig.update_yaxes(range=[1.6, 3.6],
                     mirror=True,
                     dtick=.2,
                     tickformat='.1f',
                     title=dict(text='Band gap predicted, eV',
                                font=dict(size=28),
                                ))
    fig.update_layout(width=1400, height=700,
                      template='simple_white',
                      font_size=18,
                      margin=dict(t=80),
                      )
    
    for i in range(1, 3):
        fig.update_layout({f'legend{i}': {'x': getattr(fig.layout, f'xaxis{i}')['domain'][1],
                                          'y': getattr(fig.layout, f'yaxis{i}')['domain'][0], 
                                          'xanchor': 'right',
                                          'yanchor': 'bottom',
                                          'font': {'size': 24}
                                          }})
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig



def predicted_vs_actual_plots(predicted_values: Dict[str, ArrayLike],
                              results_df: pd.DataFrame,
                              y_train_true: ArrayLike,
                              y_test_true: ArrayLike,
                              train_text: Union[ArrayLike, Sequence],
                              test_text: Union[ArrayLike, Sequence],
                              layout_params: Union[Dict, None] = None,
                              traces_params: Union[Dict, None] = None,
                              ) -> go.Figure:
    '''______________________________________________________________________________________________________________________________
    Draw Predicted vs Actual plots for all models.
    
    Parameters:
        predicted_values (Dict[str, ArrayLike]): Dictionary with predidicted values on train subset.
        results (pandas.DataFrame): Table with metrics on training and validation.
        y_train_true (ArrayLike): Vector of actual target values on train subset.
        y_test_true (ArrayLike): Vector of actual target values on test subset.
        train_text (ArrayLike | Sequence): Array with text for train points.
        test_text (ArrayLike | Sequence): Array with text for text points.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating, if None – default style.
        traces_params (Dict | None, default=None): Dictinary with parameters for test trace updating, if None – default style.


    Return:
        plotly.graph_objects.Figure: Predicted vs Actual plots as plotly.graph_objects.Figure object.
    ______________________________________________________________________________________________________________________________'''
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=list(predicted_values['train'].keys()),
                        vertical_spacing=0.15,
                        horizontal_spacing=.0675)

    for i, k in enumerate(predicted_values['train'].keys()):
        fig.add_trace(go.Scatter(x=np.arange(0, 4, 0.01),
                                 y=np.arange(0, 4, 0.01),
                                 name='Identity line',
                                 hovertemplate='',
                                 showlegend=False,
                                 mode='lines',
                                 line=dict(width=3,
                                           dash='dash',
                                           color='black',
                                           ),
                                 ),
                      row=i//4+1, col=i%4+1)
        fig.add_trace(go.Scatter(x=y_train_true,
                                 y=predicted_values['train'][k],
                                 legend='legend' + str(i+1),
                                 name='train',
                                 text=train_text,
                                 hovertemplate='train<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                                 mode='markers',
                                 marker=dict(size=7.5,
                                             symbol='square',
                                             color='#4C78A8',
                                             line_width=1.5,
                                             line_color='#0B1532',
                                             ),
                                 ),
                      row=i//4+1, col=i%4+1)
        fig.add_trace(go.Scatter(x=y_test_true,
                                 y=predicted_values['test'][k],
                                 legend='legend' + str(i+1),
                                 name='test',
                                 text=test_text,
                                 hovertemplate='test<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>%{text}',
                                 mode='markers',
                                 marker=dict(size=9.5,
                                             symbol='star',
                                             color='#E33333',
                                             line_width=1.5,
                                             line_color='#550B0B',
                                             ),
                                 ),
                      row=i//4+1, col=i%4+1)
        fig.add_annotation(go.layout.Annotation({'text': f"On test:<br>MAE = {round(results_df.loc[k, 'MAE on test'], 2):.2f}<br>RMSE = {round(results_df.loc[k, 'RMSE on test'], 2):.2f}<br>R<sup>2</sup> = {round(results_df.loc[k, 'R2 on test'], 4):.4f}",
                                                 'xref': f'x{i+1}',
                                                 'yref': f'y{i+1}',
                                                 'x': 1.95, 'y': 3.3,
                                                 'align': 'left',
                                                 'showarrow': False,
                                                 })) 
        
        fig.update_xaxes(range=[1.6, 3.6],
                         mirror=True,
                         dtick=.2,
                         tickformat='.1f',
                         title=dict(text='Band gap experimental, eV'),
                         row=i//4+1, col=i%4+1)
        fig.update_yaxes(range=[1.6, 3.6],
                         mirror=True,
                         dtick=.2,
                         tickformat='.1f',
                         title=dict(text='Band gap predicted, eV'),
                         row=i//4+1, col=i%4+1)
    

    if not isinstance(traces_params, type(None)):
        fig.update_traces(**traces_params) 
    
    fig.update_layout(width=1600, height=800,
                      template='simple_white',
                      )
    for i in range(1, 9):
        fig.update_layout({f'legend{i}': {'x': getattr(fig.layout, f"xaxis{i}")['domain'][1],
                                          'y': getattr(fig.layout, f"yaxis{i}")['domain'][0], 
                                          'xanchor': 'right',
                                          'yanchor': 'bottom',
                                          }})
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)
    
    return fig



def shap_explation(final_pipelines: Dict[str, Pipeline],
                   model_name: str,
                   X: pd.DataFrame,
                   background_size: int = 50,
                   ) -> shap.Explanation:
    '''______________________________________________________________________________________________________________________________
    Compute SHAP explanations for a trained regression model stored inside a scikit-learn pipeline.

    Parameters:
         final_pipelines (Dict[str, sklearn.pipeline.Pipeline]): Dictinary with final pipelines of trained models.
         model_name (str): Key identifying the model in ``final_pipelines``.
         X: Feature matrix used to compute SHAP values.
         background_size (int, default=50) :Number of background samples used for kernel-based explainers (e.g., SVR, GPR). Smaller
         values improve performance at the cost of approximation accuracy.

    Return:
        shap.Explanation: SHAP explanation object containing per-feature contribution values for each sample in ``X``. This object
                         can be directly passed to ``shap.plots.*`` functions.
    ______________________________________________________________________________________________________________________________'''
    if model_name not in final_pipelines.keys():
        raise ValueError("Wrong model name.")
    
    regressor = final_pipelines[model_name].named_steps['model']
    scaler = final_pipelines[model_name].named_steps['scaler']
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    X_scaled = X_scaled.rename(columns={k: v.replace('<sub>', '_').replace('</sub>', '').replace('<sup>', '^').replace('</sup>', '') for k,v in visualization_descriptor_names.items()})

    if model_name in ['RandomForest', 'ExtraTrees', 'XGBoost']:
        explainer = shap.TreeExplainer(regressor)
    elif model_name in ['Linear', 'ElasticNet', 'GLR', 'Huber', 'BayesianRidge', 'PLC']:
        explainer = shap.LinearExplainer(regressor, X_scaled)
    elif model_name in ['SVR', 'GPR']:
        background = shap.sample(X_scaled, min(background_size, len(X_scaled)))
        explainer = shap.KernelExplainer(regressor.predict, background)
    else:
        print('Use custom code.')
        return None
    
    shap_values = explainer(X_scaled)

    return shap_values



def shap_bar_plot(shap_values: shap.Explanation,
                  xlim_max: int,
                  max_display: int = 20,
                  text_shift: float = 3.75,
                  x_ticklabels_scale: float = 1.225,
                  decimal_places: int = 4,
                  change_colors: bool = False,
                  new_colors: Sequence[str] = ['#EE0510', '#053A61'],
                  ) -> None:
    '''______________________________________________________________________________________________________________________________
    Draw feature importance bar plot by SHAP values.

    Parameters:
        shap_values (shap.Explanation): SHAP explanation object containing per-feature contribution values for each sample in ``X``.
        max_display (int, default=20): Number of the first most important descriptors for displaying.
        xlim_max (int): Maximum of SHAP value (on X axis) in hundredfold scale.
        text_shift (float, default=3.75): Value to shift text for each plot bin in hundredfold scale of X axis.
        x_ticklabels_scale (float, default=1.225): Parameter for Y axis labels position (fraction from X axis).
        decimal_places (int, default=4): Number of digits after decimal for bin text.
        change_colors (bool, default=False): Flag to change default shap-librery colors.
        new_colors (Sequence[str], default=['#EE0510', '#053A61']): Array with new colors for ones changing (shape = 2, for positive
                                                                   and negative values).

    Return:
        None:
    ______________________________________________________________________________________________________________________________'''
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    positive_color, negative_color = new_colors

    shap.plots.bar(shap_values*100, max_display=max_display, show=False)
    ax = plt.gca()
    for txt in ax.findobj(matplotlib.text.Text)[::-1]:
        if '+' in txt.get_text():
            txt.set_text(f"+{float(txt.get_text())/100:.{decimal_places}f}")
            txt.set_position((txt.get_position()[0]+text_shift, txt.get_position()[1]))
            
        elif txt.get_text() == 'mean(|__mul__(SHAP value)|)':
            txt.set_text('Mean Absolute SHAP value')

    if change_colors:
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children()[:-1]:
                if (isinstance(fcc, matplotlib.patches.Rectangle)):
                    if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                        fcc.set_facecolor(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                        fcc.set_color(negative_color)
                elif (isinstance(fcc, plt.Text)):
                    if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                        fcc.set_color(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                        fcc.set_color(negative_color)
    
    ax.set_xlim(xlim_max+1, 0)
    ax.set_xticks(ax.get_xticks(), labels=[0]+list(np.array(ax.get_xticks()[1:])/100))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='y', which='major', length=0)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('center')
        label.set_x(x_ticklabels_scale)



def model_based_optimization(pipline: Pipeline,
                             feature_names: Union[ArrayLike, Sequence],
                             bounds_x: Dict[str, Tuple],
                             initial_guesses: Union[ArrayLike, Sequence, None] = None,
                             n_trials: int = 2500,
                             seed: int = 7,
                             ) -> Tuple[pd.DataFrame, Dict, float]:
    '''______________________________________________________________________________________________________________________________
    Perform model-based (surrogate) optimization using a trained pipeline as the predictive model. The function uses Optuna to
    perform Bayesian optimization over the descriptor space to identify candidate feature combinations minimizing the predicted
    target.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): Fitted scikit-learn pipeline containing at least a 'model' step and  optionally a 
                                             'scaler' step for feature preprocessing.
        feature_names (ArrayLike | Sequence): List or array of feature names to be optimized. Only features included here are
                                             considered in the optimization.
        bounds_x (Dict[str, Tuple]): Dictionary specifying the lower and upper bounds for each feature. Keys correspond to feature
                                    names, values are tuples of the form (low, high).
        initial_guesses (ArrayLike | Sequence | None, default=None): Array of initial points for model based optimization with
                                                                    selected configuration of descriptor space.
        n_trials (int, default=2500): Number of Bayesian optimization trials to perform.
        seed (int, default=7): Random seed for reproducibility of the optimization.

    Returns:
        history_df (pandas.DataFrame): DataFrame containing the history of all optimization steps. Columns include:
                                          - 'iter': iteration number;
                                          - 'value': predicted target for that step;
                                          - feature columns with the suggested values for each iteration.
        best_x (Dict): Dictionary of feature values corresponding to the minimal predicted target found during the optimization.
        best_value (float): Minimal predicted target value obtained during optimization.
    ______________________________________________________________________________________________________________________________'''
    history = []

    def history_callback(study, trial):
        '''______________________________________________________________________________________________________________________________
        Callback function for Optuna optimazer. Save trial history to dictionary.

        Parameters:
            study (optuna.study.study.Study): Optuna Study object corresponds to an optimization task,
            trial (optuna.trial.Trial): Optuna trial object used to sample hyperparameters.
        
        Return:
            None
        ______________________________________________________________________________________________________________________________'''
        history.append({"iter": trial.number, "value": trial.value, **trial.params})

    def objective_inverse(trial) -> float:
        '''______________________________________________________________________________________________________________________________
        Optuna objective function for model-based optimization.

        Parameters:
            trial (optuna.trial.Trial): Optuna trial object used to sample hyperparameters.
        
        Return:
            float: Prediction of model.
        ______________________________________________________________________________________________________________________________'''
        x = {name: trial.suggest_float(name, low, high) for name, (low, high) in bounds_x.items() if name in feature_names}

        X_candidate = pd.DataFrame([x])
        X_scaled = pipline.named_steps['scaler'].transform(X_candidate)

        return pipline.named_steps['model'].predict(X_scaled)[0]

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))

    if not isinstance(initial_guesses, type(None)):
        for params in initial_guesses:
            study.enqueue_trial(params)

    study.optimize(objective_inverse, n_trials=n_trials, callbacks=[history_callback])

    history_df = pd.DataFrame(history)
    best_x = study.best_params
    best_value = study.best_value

    return history_df, best_x, best_value




def optimization_convergence_plot(history_df: pd.DataFrame,
                                  layout_params: Union[Dict, None] = None,
                                  traces_params: Union[Dict, None] = None,
                                  ) -> go.Figure:
    '''______________________________________________________________________________________________________________________________
    Draw convergence plot for optimization process.

    Parameters:
        history_df (pandas.DataFrame): DataFrame containing the history of all optimization steps. Columns include:
                                      - 'iter': iteration number;
                                      - 'value': predicted target for that step.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating, if None – default style.
        traces_params (Dict | None, default=None): Dictinary with parameters for test trace updating, if None – default style.
    
    Return:
        plotly.graph_objects.Figure: Convergence plot as plotly.graph_objects.Figure object.
    ______________________________________________________________________________________________________________________________'''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df['iter'],
                             y=history_df['value'],
                             name='Value on iteration',
                             mode='markers',
                             showlegend=False,
                             hovertemplate='Iteration %{x:.2f}<br>Value: %{y:.2f}',
                             marker=dict(line_width=1),
                             ))
    fig.add_trace(go.Scatter(x=history_df['iter'],
                             y=history_df['value'].expanding().mean(),
                             name='Mean value',
                             hovertemplate='Iteration %{x:.2f}<br>Mean value: %{y:.2f}',
                             line=dict(width=5,
                                      color='#00CC00',
                                      ),
                             ))
    fig.add_trace(go.Scatter(x=history_df['iter'],
                             y=history_df['value'].cummin(),
                             name='Minimal value',
                             hovertemplate='Iteration %{x:.2f}<br>Minimal value: %{y:.2f}',
                             line=dict(width=4,
                                       dash='dashdot',
                                       color='#EE0000',
                                       ),
                             ))
    
    if not isinstance(traces_params, type(None)):
        fig.update_traces(**traces_params) 

    fig.update_xaxes(range=[-10, 1000],
                     tickformat='.0f', 
                     dtick=1000,
                     mirror=True,
                     title=dict(text='Iteration number',
                                font=dict(size=24),
                                ))
    fig.update_yaxes(range=[1.65, 3.25],
                     tickformat='.1f',
                     mirror=True,
                     title=dict(text='Band gap, eV',
                                font=dict(size=24),
                                ))
    fig.update_layout(width=1200, height=600,
                      template='simple_white',
                      legend=dict(x=.825, y=.55,
                                  font=dict(size=18),
                                  ),
                      margin=dict(t=50),
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)

    return fig      



def repeated_hold_out_validation(df_raw: pd.DataFrame,
                                 descritor_spaces: Union[ArrayLike, Sequence],
                                 hyperparameters_dicts: Sequence[Dict[str, Dict]],
                                 split_seeds: Union[Sequence, ArrayLike, Iterable] = range(105),
                                 seed: int = 7,
                                 ) -> Tuple[pd.DataFrame]:
    '''______________________________________________________________________________________________________________________________
    Perform repeated hold-out validation of trained models on stratified train–test splits of all datasate. Make LOO-CV on train
    subset and prediction on test subset in each split.
    
    Parameters:
        df_raw (pandas.DataFrame): Original prepared dataset.
        descritor_spaces (ArrayLike | Sequence): Array of feature names for each descriptor space.
        hyperparameters_dicts (Sequence[Dict[str, Dict]]): Array of dictionaries with hyperparameters for each studied model for each
                                                          descriptor space.
        split_seeds (Sequence | ArrayLike | Iterable, default=range(105)): Set of split random states.
        seed (int, default=7): Random seed for reproducibility.

    Return:
        pandas.DataFrame: DataFrames containing metrics on validation and testing on each train–test split (number of DataFrames is
                         equal to number of descritor spaces).
    ______________________________________________________________________________________________________________________________'''
    result_list = [{} for _ in range(len(descritor_spaces))]

    for num, i in enumerate(split_seeds, 1):
        print('________________________________________')
        print(f'Split number {num}')
        counts = df_raw['Composition'].value_counts()
        rare = counts[counts < 3].index

        df_rare = df_raw[df_raw['Composition'].isin(rare)]
        df_common = df_raw[~df_raw['Composition'].isin(rare)]

        X_common = df_common.drop(columns=['Refcode', 'Bandgap', 'Composition'])
        y_common = df_common['Bandgap']

        X_train_c_new, X_test_new, y_train_c_new, y_test_new = train_test_split(X_common, y_common,
                                                                                test_size=0.15,
                                                                                stratify=df_common['Composition'],
                                                                                random_state=i)

        X_train_new = pd.concat([X_train_c_new, df_rare.drop(columns=['Refcode', 'Bandgap', 'Composition'])])
        y_train_new = pd.concat([y_train_c_new, df_rare['Bandgap']])

        for j, (descriptor_space, hyperparameters_dict)  in enumerate(zip(descritor_spaces, hyperparameters_dicts)):
            X_train_new_i = X_train_new[descriptor_space]
            final_pipelines_i, predicted_values_i, results_i = train_and_validate(hyperparameters_dict=hyperparameters_dict, X_train=X_train_new_i, y_train=y_train_new, seed=seed)
            predicted_values_i, results_i = predict_models(final_pipelines=final_pipelines_i, predicted_values=predicted_values_i, results=results_i, X_train=X_train_new_i, X_test=X_test_new, y_test=y_test_new)
            result_list[j][i] = results_i
            
        print('________________________________________')
        print()
    
    for k in range(len(result_list)):
        result_list[k] = pd.DataFrame([{'ind': ind,
                                        'model': model,
                                        'metric': key.split(' on ')[0],
                                        'offset': key.split(' on ')[1],
                                        'value': value,} for ind, models in result_list[k].items() for model, metrics in models.items() for key, value in metrics.items() if key.split(' on ')[1] != 'train'])
    
    return tuple(result_list)



def repeated_hold_out_validation_result_plot(rhov_result_df: pd.DataFrame,
                                             model_names: Union[ArrayLike, Sequence],
                                             layout_params: Union[Dict, None] = None,
                                             traces_params: Union[Dict, None] = None,
                                             ) -> go.Figure:
    '''_____________________________________________________________________________________________________________________________
    Draw boxplots for repeated hold out validation results.
    
    Parameters:
        rhov_result_df (pandas.DataFrame): DataFrames containing metrics on validation and testing on each train–test split
                                               from result of repeated hold out validation
        model_names (ArrayLike | Sequence): Array of model names.
        layout_params (Dict | None, default=None): Dictinary with parameters for layout updating, if None – default style.
        traces_params (Dict | None, default=None): Dictinary with parameters for test trace updating, if None – default style.
    
    Return:
         plotly.graph_objects.Figure: Boxplots for repeated hold out validation results.
    _____________________________________________________________________________________________________________________________'''
    fig = make_subplots(rows=3, cols=1,
                           vertical_spacing=0.1)
    c = np.array([['hsl('+str(h)+',30%'+',30%)', 'hsl('+str(h)+',50%'+',50%)'] for h in np.linspace(0, 360, 10)]).ravel()

    offsets_pos = {"validation": -0.25, "test": 0.25}


    for i, metric in enumerate(['MAE', 'RMSE', 'R2'], 1):
        df_metric = rhov_result_df.query(f"metric == '{metric}'")
        for j, (model, offset) in enumerate(product(rhov_result_df['model'].unique(), rhov_result_df['offset'].unique())):
            sub = df_metric[(df_metric["model"] == model) & (df_metric["offset"] == offset)]
            fig.add_trace(go.Box(y=sub["value"],
                                 x=[list(model_names).index(model) + offsets_pos[offset]] * len(sub["value"]),  # уникально
                                 name=offset,
                                 marker_color=c[j],
                                 showlegend=True if j in [0, 1] else False,
                                 legend='legend'+str(i),
                                 boxmean='sd',
                                 width=.4,
                                 ),
                          row=i, col=1)

    if not isinstance(traces_params, type(None)):
        fig.update_traces(**traces_params) 

    fig.update_xaxes(tickvals=list(range(8)),
                     ticktext=[f"<span style='color:{color}'>{m}</span>" for m, color in zip(model_names, ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 10)])],
                     tickfont=dict(size=20),
                     tickangle=45,
                     )
    fig.update_yaxes(title=dict(text='MAE, eV',
                                font=dict(size=24),
                                ),
                     tickformat='.2f',
                     row=1, col=1)
    fig.update_yaxes(title=dict(text='RMSE, eV',
                                font=dict(size=24),
                                ),
                     tickformat='.2f',
                     row=2, col=1)
    fig.update_yaxes(title=dict(text='R<sup>2</sup>',
                                font=dict(size=24),
                                ),
                     tickformat='.2f',
                     row=3, col=1)

    fig.update_layout(width=1200, height=1800,
                      template='plotly_white',
                      boxmode='group',
                      legend1=dict(x=1,
                                   font=dict(size=24),
                                   ),
                      legend2=dict(x=1, y=0.625,
                                   font=dict(size=24),
                                   ),
                      legend3=dict(x=1, y=0.225,
                                   font=dict(size=24),
                                   ),
                      )
    
    if not isinstance(layout_params, type(None)):
        fig.update_layout(**layout_params)
    
    return fig



def save_all_model_piplines(final_pipelines: Dict[str, Pipeline],
                            path_name: str,
                            ) -> None:
    '''________________________________________________________________________________________________________________________________
     Save all trained model pipelines in binary format by pickle library.

     Parameters:
            final_pipelines (Dict [str, sklearn.pipeline.Pipeline]): Dictinory with all trained model pipelines.
            path_names (str): Path to saving and file name without extension ".pickle" or ".pkl".

     Return:
           None:
     ________________________________________________________________________________________________________________________________'''
    with open(path_name+'.pickle', 'wb') as file:
        pickle.dump(final_pipelines, file, protocol=pickle.HIGHEST_PROTOCOL)



def save_model_and_pipeline(model_pipeline: Pipeline, 
                            path_name: str,
                            ) -> None:
     '''________________________________________________________________________________________________________________________________
     Save trained model pipeline in binary format by pickle library.

     Parameters:
            model_pipeline (sklearn.pipeline.Pipeline): Trained model pipeline.
            path_names (str): Path to saving and file name without extension ".pickle" or ".pkl".

     Return:
           None:
     ________________________________________________________________________________________________________________________________'''
     with open(path_name+'.pickle', 'wb') as file:
          pickle.dump(model_pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)



########################################################################################################################################



######################################################## CODE FOR MODEL APPLYING #######################################################

def load_all_model_pipelines(path_file_name: str,
                             ) -> Pipeline:
     '''________________________________________________________________________________________________________________________________
     Load all saving pretrained model pipelines from binary pickle format.

     Parameters:
            path_file_names (str): Path to file and file name.

     Return:
           Dict[str, sklearn.pipeline.Pipeline]: Dictinory with all pretrained model pipelines.
     ________________________________________________________________________________________________________________________________'''
     with open(path_file_name, 'rb') as file:
          return pickle.load(file)
  


def load_model_pipeline(path_file_name: str,
                        ) -> Pipeline:
     '''________________________________________________________________________________________________________________________________
     Load saving pretrained model pipeline from binary pickle format.

     Parameters:
            path_file_names (str): Path to file and file name.

     Return:
           sklearn.pipeline.Pipeline: Pretrained model pipeline.
     ________________________________________________________________________________________________________________________________'''
     with open(path_file_name, 'rb') as file:
          return pickle.load(file)



def get_geometrical_descriptors(cif: str,
                                M: str,
                                X: str,
                                length_of_search: float = 4.0
                                ) -> pd.Series:
    '''________________________________________________________________________________________________________________________________
    Calculate Geometrical descriptors for one crystal structure from CIF by PayMatGen.
    PayMatGen works unstable, so results are required to check!!!

    Parameters:
        cif (str): CIF name with path.
        M (str): metal composition: "Bi" or "Sb".
        X (str): halogen composition: "I", "Br" or "Cl".
        length_of_search (float, default=4.0): maximal M–X bond length in A.

    Return:
        pd.Series: array with anion gemetry descriptors.
    ________________________________________________________________________________________________________________________________'''


    
    def extract_digits(s):
        match = re.search(r'\D*(\d+)', s)  # Ищем первую последовательность цифр, включая предшествующие буквы
        if match:
            return match.group()  # Возвращаем буквы перед цифрами и сами цифры
        return ""  # Если цифр нет, возвращаем пустую строку
    
    def get_descriptors_for_one_M_X_label(M_label: str, X_label: str) -> pd.DataFrame:
        '''
        Calculate M–X distances, X–M–X angles and octahedra distortion degree parameters: ∆d and sigma^, for ONE octahedron
        '''
        def angle_search(que: str, answ: str, min_max= max) -> None:
            ''' 
            To help
            '''
            angles_aid = {}
            for ind in dict_ind_distance.keys():
                angle = structure.get_angle(inds['X_'+que+'_ind'], M_ind, ind)
                angles_aid[ind] = angle
            for ind, angle in angles_aid.items():
                if angle  == min_max(angles_aid.values()):
                    inds['X_'+answ+'_ind'] = ind
                    distances['X_'+answ] = round(dict_ind_distance[ind], 4)
                    del dict_ind_distance[ind]
                    break
            del angles_aid

        sites_m = [site for site in structure._sites if site._label == M_label]
        res = []
        for i in range(len(sites_m)):
            M_ind = structure.index(sites_m[i])
            
            # Searsh of Xogen neighbors for M(III)
            dict_ind_distance = {site.index: site.nn_distance for site in structure.get_neighbors(structure[M_ind], r=length_of_search) if X_label in site._label}
            
            distances, angles, inds = {}, {}, {}

            # Search X-t1
            for ind, distance in dict_ind_distance.items():
                if distance == min(dict_ind_distance.values()):
                    inds['X_t1_ind'] = ind
                    distances['X_t1'] = round(dict_ind_distance[ind], 4)
                    del dict_ind_distance[ind]
                    break

            # Search X-d3
            angle_search('t1', 'd3')
            
            # Search X-t2
            number_of_X_neighbor = {ind: len([neighbor for neighbor in structure.get_neighbors(structure[ind], r=length_of_search) if M in neighbor._label and neighbor.index != inds['X_t1_ind']]) for ind in dict_ind_distance.keys()}
            for ind, n in number_of_X_neighbor.items():
                if n == min(number_of_X_neighbor.values()):
                    inds['X_t2_ind'] = ind
                    distances['X_t2'] = round(dict_ind_distance[ind], 4)
                    del dict_ind_distance[ind]
                    break
            del number_of_X_neighbor
            
            # Search X-d1
            angle_search('t2', 'd1')
            
            # Search X-d2 and X-d4
            angle_search('d1', 'd2', min_max=min)
            inds['X_d4_ind'] = list(dict_ind_distance.keys())[0]
            distances['X_d4'] = round(dict_ind_distance[list(dict_ind_distance.keys())[0]], 4)
            del dict_ind_distance
            
            # Calculate angles:
            angles['t1_t2'] = round(structure.get_angle(inds['X_t1_ind'], M_ind, inds['X_t2_ind']), 3)
            angles['t1_d1'] = round(structure.get_angle(inds['X_t1_ind'], M_ind, inds['X_d1_ind']), 3)
            angles['t1_d2'] = round(structure.get_angle(inds['X_t1_ind'], M_ind, inds['X_d2_ind']), 3)
            angles['t1_d3'] = round(structure.get_angle(inds['X_t1_ind'], M_ind, inds['X_d3_ind']), 3)
            angles['t1_d4'] = round(structure.get_angle(inds['X_t1_ind'], M_ind, inds['X_d4_ind']), 3)
            angles['t2_d1'] = round(structure.get_angle(inds['X_t2_ind'], M_ind, inds['X_d1_ind']), 3)
            angles['t2_d2'] = round(structure.get_angle(inds['X_t2_ind'], M_ind, inds['X_d2_ind']), 3)
            angles['t2_d3'] = round(structure.get_angle(inds['X_t2_ind'], M_ind, inds['X_d3_ind']), 3)
            angles['t2_d4'] = round(structure.get_angle(inds['X_t2_ind'], M_ind, inds['X_d4_ind']), 3)
            angles['d1_d2'] = round(structure.get_angle(inds['X_d1_ind'], M_ind, inds['X_d2_ind']), 3)
            angles['d1_d3'] = round(structure.get_angle(inds['X_d1_ind'], M_ind, inds['X_d3_ind']), 3)
            angles['d1_d4'] = round(structure.get_angle(inds['X_d1_ind'], M_ind, inds['X_d4_ind']), 3)
            angles['d2_d3'] = round(structure.get_angle(inds['X_d2_ind'], M_ind, inds['X_d3_ind']), 3)
            angles['d2_d4'] = round(structure.get_angle(inds['X_d2_ind'], M_ind, inds['X_d4_ind']), 3)
            angles['d3_d4'] = round(structure.get_angle(inds['X_d3_ind'], M_ind, inds['X_d4_ind']), 3)
            del inds
            
            lengths = np.array(list(distances.values()))
            angles_90 = np.array(sorted(angles.values(), key=lambda x: np.abs(x - 90))[:12])
            d_average = np.mean(lengths)
            delta_d = 1/6*np.sum(((lengths-np.mean(lengths))/np.mean(lengths))**2)
            sigma_2 = 1/11*(np.sum((angles_90-90)**2))
            
            res.append(pd.concat([pd.Series(distances), pd.Series(angles), pd.Series({'d_average': d_average}), pd.Series({'delta_d': delta_d, 'sigma_2': sigma_2})]))
        best_res = pd.DataFrame(res)
        best_res = best_res[['X_t1', 'X_t2', 'X_d1', 'X_d2', 'X_d3', 'X_d4']+best_res.columns[6:].to_list()]
        return best_res[best_res['sigma_2'] == best_res['sigma_2'].min()].iloc[0]

    # Read CIF and make superstructure:
    structure = CifParser(cif, occupancy_tolerance=1000).get_structures()[0]
    structure.make_supercell([3, 3, 3])
    
    M_sites = sorted(set([extract_digits(site._label) for site in structure._sites if site._label[:2] in M.split()]))
    X_sites = X.split()
    if len(M_sites) == len(X_sites) == 1: # Calculate for one crystal independent octahedron
        return get_descriptors_for_one_M_X_label(M_sites[0], X_sites[0])
    elif len(M_sites) < 1:
        print("No structure data")
    else:
        octahedra_params_list = []
        for M_site in M_sites:
            for X_site in X_sites:
                octahedra_params_list.append(get_descriptors_for_one_M_X_label(M_site, X_site))
        return pd.DataFrame(octahedra_params_list).mean()



def get_descriptors_one_structure(cif_name_and_path: str,
                                  T: float,
                                  m: str,
                                  x: str,
                                  xx_contacts_string: str,
                                  max_mx_bond_length: float = 4.0
                                  ) -> pd.Series:
    '''________________________________________________________________________________________________________________________________
    Calculate all descriptors for one crystal structure from CIF.
    PayMatGen works unstable, so results are required to check!!!

    Parameters:
        cif_name_and_path (str): CIF name with path.
        T (float): X-ray difraction experiment temperature.
        M (str): metal composition: "Bi" or "Sb".
        X (str): halogen composition: "I", "Br" or "Cl".
        xx_contacts_string (str): space-separated string with X···X contact distanced originated from one [MX6] octahedron.
        max_mx_bond_length (float, default=4.0): maximal M–X bond length in A.

    Return:
        pd.Series: array with all descriptors.
    _______________________________________________________________________________________________________________________________'''
    r_vdw = {'I': 2.1, 'Br': 1.85, 'Cl': 1.8}
    descriptors = get_geometrical_descriptors(cif_name_and_path, M=m, X=x, length_of_search=max_mx_bond_length)
    descriptors = pd.concat([pd.Series({'Temperature': T}), descriptors])
    xx_contacts = np.array([float(cont) for cont in xx_contacts_string.split()])
    xx_vdw = np.array(list(filter(lambda d: d <= 2 * r_vdw[x], xx_contacts)))
    n_xx = len(xx_contacts)
    descriptors['N_XX'] = n_xx
    n_vdw = len(xx_vdw)
    descriptors['N_VdW'] = n_vdw
    xx_min = xx_contacts.min()
    descriptors['XX_min'] = xx_min
    xx_aver = xx_contacts.mean()
    descriptors['XX_average'] = xx_aver
    xx_vdw_aver = xx_vdw.mean()
    descriptors['VdW_average'] = xx_vdw_aver
    descriptors['XXmin_2r'] = xx_min/(2 * r_vdw[x])
    descriptors['XXaver_2r'] = xx_aver/(2 * r_vdw[x])
    descriptors['VdW_2r'] = xx_vdw_aver/(2 * r_vdw[x])
    descriptors['M'] = {'Bi': 0, 'Sb': 1}[m]
    descriptors['I'] = 0
    descriptors['Br'] = 0
    descriptors['Cl']  = 0
    descriptors[x] = 1
    return descriptors



def get_descriptors_all_structures(cifs_array,
                                   t_array,
                                   m_array,
                                   x_array,
                                   xx_contacts_array,
                                   max_mx_bond_length: float = 4.0
                                   ) -> pd.DataFrame:
    '''________________________________________________________________________________________________________________________________
    Calculate all descriptors for set of crystal structures from CIFs.
    PayMatGen works unstable, so results are required to check!!!

    Parameters:
        cifs_array: Array with CIF names and paths for each structure.
        t_array (ArrayLike): Array with X-ray difraction experiment temperatures for each structure.
        M: Array with metal compositions for each structure.
        X: Array with halogen compositions for each structure.
        xx_contacts_string: Array with space-separated string with X···X contact distanced originated from one [MX6]
                                             octahedron for each structure.
        max_mx_bond_length (float, default=4.0): maximal M–X bond length in A.

    Return:
        pd.DataFrame: dataframe for band gap prediction.
    _______________________________________________________________________________________________________________________________'''
    return pd.DataFrame([get_descriptors_one_structure(cif_name_and_path=cif, T=t, m=m, x=x, xx_contacts_string=xx_str, max_mx_bond_length=max_mx_bond_length) for cif, t, m, x, xx_str in zip(cifs_array, t_array, m_array, x_array, xx_contacts_array)])

########################################################################################################################################