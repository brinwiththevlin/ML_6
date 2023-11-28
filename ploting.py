import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from typing import List


def plot_error(error_vect: List[float], style: str, lrate: float):
    """plots error curve over time (epochs) based on training style and learning rate

    Args:
        error_vect (List[float]): list containing absolute error at each time step
        style (str): batch or incremental
        lrate (float): learning rate
    """
    fig = px.line(x=list(range(1, 101)), y=error_vect)
    fig.update_layout(
        xaxis_title="epoch",
        yaxis_title="absolute error",
        title=f"absolute error against time ({style}, lrate={lrate})",
    )
    lrate_str = str(lrate)[2:]
    # fig.show()
    fig.write_html(f"pics/abs_error_{style}_{lrate_str}.html")


def plot_decision_planes(X, Y, planes, style, lrate):
    positive = X[Y == 1]
    negative = X[Y == 0]
    x_vals = [i / 2 for i in range(-100, 101)]

    for plane, t in enumerate([5, 10, 50, 100]):
        y_vals = _decision_line(planes[plane])
        lrate_str = str(lrate)[2:]

        fig = go.Figure()

        # Add plus markers
        fig.add_trace(
            go.Scatter(
                x=positive[:, 0],
                y=positive[:, 1],
                mode="markers",
                marker=dict(symbol="circle", color="blue"),
                name="possitive",
            )
        )

        # Add minus markers
        fig.add_trace(
            go.Scatter(
                x=negative[:, 0],
                y=negative[:, 1],
                mode="markers",
                marker=dict(symbol="square", color="red"),
                name="negative",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color="black", width=2),
                name="Decision Boundary",
            )
        )

        fig.update_layout(
            title=f"Decision Boundary Plot: {style} ({lrate}, epoch: {t})",
            xaxis=dict(title="X1"),
            yaxis=dict(title="X2"),
            showlegend=True,
        )
        # fig.show()

        fig.write_html(f"pics/decision_boundary_{style}_{lrate_str}_{t}.html")


def _decision_line(weights):
    y = [-(weights[1] * (i / 2) + weights[0]) / weights[2] for i in range(-100, 101)]
    return y
