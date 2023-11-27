import plotly.express as px


def plot_error(error_vect):
    fig = px.line(x=list(range(1, 101)), y=error_vect)
    fig.show()
    pass
