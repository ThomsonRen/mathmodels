import plotly.graph_objects as go
import pandas as pd

df = pd.read_excel('../_static/lecture_specific/evaluation_model/COMAP_RollerCoasterData_2018.xlsx')
df = df.head(10)

c_list = 


v = []

for c in list(df.columns):
    v.append(df[c])

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=v,
               fill_color='lavender',
               align='left'))
])
fig.show()



