import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


st.sidebar.title("About")
st.sidebar.info("This is a demo application written to demonstrate basic visualizations created with "
                "streamlit and plotly")
st.sidebar.subheader("Choose the visualization")
imageselect = st.sidebar.selectbox("",
                                   ['Linear function', 'Quadratic function',
                                    'Normal distribution', 'Binomial distribution',
                                    ])



x_list = [i for i in range(-100, 100)]
y_list = []

if imageselect == 'Linear function':
    st.subheader("Linear function")
    a_lin = int(st.text_input("Parameter a:", 1))
    b_lin = int(st.text_input("Parameter b:", 1))
    for x in x_list:
        y_list.append(a_lin * x + b_lin)
    chart_data = pd.DataFrame(y_list, x_list)
    fig = px.line(chart_data)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[-100, 100], title_text='Y')
    fig.update_xaxes(range=[-100, 100], title_text='X')
    st.write(fig)

elif imageselect == 'Quadratic function':
    st.subheader("Quadratic function")
    a_quad = int(st.text_input("Parameter a :", 1))
    b_quad = int(st.text_input("Parameter b :", 2))
    c_quad = int(st.text_input("Parameter c:", 4))
    for x in x_list:
        y_list.append(a_quad * x ** 2 + b_quad * x + c_quad)
    chart_data = pd.DataFrame(y_list, x_list)
    fig = px.line(chart_data)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[-100, 100], title_text='Y')
    fig.update_xaxes(range=[-100, 100], title_text='X')
    st.write(fig)

elif imageselect == 'Normal distribution':
    st.subheader("Normal distribution")
    n_norm = int(st.text_input("N:", 1000))
    mu = float(st.text_input("Mean:", 1))
    std = float(st.text_input("Standard deviation:", 2))
    rand_var = np.random.normal(mu, std, n_norm)
    fig = ff.create_distplot([rand_var], ['Random variable'])
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 1], title_text='Probability')
    fig.update_xaxes(range=[mu - 4*std, mu + 4*std], title_text='X')
    st.write(fig)

elif imageselect == 'Binomial distribution':
    st.subheader("Binomial distribution")
    n_bin = int(st.text_input("n:", 1000))
    p = float(st.text_input("p (0.xxx):", 0.5))
    rand_var = np.random.binomial(n_bin, p, n_bin)
    fig = ff.create_distplot([rand_var], ['Random variable'])
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 1], title_text= 'Probability')
    fig.update_xaxes(range=[0, n_bin], title_text = 'N Successful')
    st.write(fig)







