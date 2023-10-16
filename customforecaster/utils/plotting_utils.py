from itertools import cycle

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf


def two_line_plot_secondary_axis(
    x,
    y1,
    y2,
    y1_name="y1",
    y2_name="y2",
    title="",
    legends=None,
    xlabel="Time",
    ylabel="Value",
    dash_secondary=False,
):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1_name),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=y2, name=y2_name, line=dict(dash="dash") if dash_secondary else None
        ),
        secondary_y=True,
    )
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        title_text=title,
        titlefont={"size": 20},
        legend_title=None,
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=12),
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=12),
        ),
    )
    return fig


def format_plot(
    fig,
    legends=None,
    xlabel="Time",
    ylabel="Value",
    figsize=(500, 900),
    font_size=15,
    title_font_size=20,
):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=figsize[1],
        height=figsize[0],
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
    )
    return fig


def plot_autocorrelation(series, vertical=False, figsize=(500, 900), **kwargs):
    if "qstat" in kwargs.keys():
        warnings.warn("`qstat` for acf is ignored as it has no impact on the plots")
        kwargs.pop("qstat")
    acf_args = ["adjusted", "nlags", "fft", "alpha", "missing"]
    pacf_args = ["nlags", "method", "alpha"]
    if "nlags" not in kwargs.keys():
        nobs = len(series)
        kwargs["nlags"] = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    kwargs["fft"] = True
    acf_kwargs = {k: v for k, v in kwargs.items() if k in acf_args}
    pacf_kwargs = {k: v for k, v in kwargs.items() if k in pacf_args}
    acf_array = acf(series, **acf_kwargs)
    pacf_array = pacf(series, **pacf_kwargs)
    is_interval = False
    if "alpha" in kwargs.keys():
        acf_array, _ = acf_array
        pacf_array, _ = pacf_array
    x_ = np.arange(1, len(acf_array))
    rows, columns = (2, 1) if vertical else (1, 2)
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"],
    )
    # ACF
    row, column = 1, 1
    [
        fig.append_trace(
            go.Scatter(
                x=(x, x), y=(0, acf_array[x]), mode="lines", line_color="#3f3f3f"
            ),
            row=row,
            col=column,
        )
        for x in range(1, len(acf_array))
    ]
    fig.append_trace(
        go.Scatter(
            x=x_, y=acf_array[1:], mode="markers", marker_color="#1f77b4", marker_size=8
        ),
        row=row,
        col=column,
    )
    # PACF
    row, column = (2, 1) if vertical else (1, 2)
    [
        fig.append_trace(
            go.Scatter(
                x=(x, x), y=(0, pacf_array[x]), mode="lines", line_color="#3f3f3f"
            ),
            row=row,
            col=column,
        )
        for x in range(1, len(pacf_array))
    ]
    fig.append_trace(
        go.Scatter(
            x=x_,
            y=pacf_array[1:],
            mode="markers",
            marker_color="#1f77b4",
            marker_size=8,
        ),
        row=row,
        col=column,
    )
    fig.update_traces(showlegend=False)
    fig.update_yaxes(zerolinecolor="#000000")
    fig.update_layout(
        autosize=False,
        width=figsize[1],
        height=figsize[0],
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        yaxis=dict(
            titlefont=dict(size=12),
        ),
        xaxis=dict(
            titlefont=dict(size=12),
        ),
    )
    return fig


def plot_outliers(x, ts, outlier_mask, method, font_size=15):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=ts, mode="lines", name="Original"))
    fig.add_trace(
        go.Scatter(
            x=x[outlier_mask],
            y=ts[outlier_mask],
            mode="markers",
            marker_symbol="star",
            marker_size=5,
            name="Outliers",
        )
    )
    fig.update_layout(
        title_text=f"Outliers using {method}: # of Outliers: {outlier_mask.sum()} | % of Outliers: {outlier_mask.sum()/len(ts)*100:.2f}%",
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
    )
    return fig


def plot_forecast(
    pred_df, forecast_columns, target_column, forecast_display_names=None
):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)
    mask = ~pred_df[forecast_columns[0]].isnull()
    # colors = ["rgba("+",".join([str(c) for c in plotting_utils.hex_to_rgb(c)])+",<alpha>)" for c in px.colors.qualitative.Plotly]
    colors = [
        c.replace("rgb", "rgba").replace(")", ", <alpha>)")
        for c in px.colors.qualitative.Dark2
    ]
    # colors = [c.replace("rgb", "rgba").replace(")", ", <alpha>)") for c in px.colors.qualitative.Safe]
    act_color = colors[0]
    colors = cycle(colors[1:])
    dash_types = cycle(["dash", "dot", "dashdot"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df[target_column],
            mode="lines",
            line=dict(color=act_color.replace("<alpha>", "0.3")),
            name="Actual Consumption",
        )
    )
    for col, display_col in zip(forecast_columns, forecast_display_names):
        fig.add_trace(
            go.Scatter(
                x=pred_df[mask].index,
                y=pred_df.loc[mask, col],
                mode="lines",
                line=dict(
                    dash=next(dash_types), color=next(colors).replace("<alpha>", "1")
                ),
                name=display_col,
            )
        )
    return fig
