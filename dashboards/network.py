import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import matplotlib.colors as mcolors

def main():
    # Create the graph
    df = pd.read_csv('data/witcher.csv', index_col=0)

    df.drop('Type', axis=1, inplace=True)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    # Set node positions
    pos = nx.spring_layout(G, k=0.3)

    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values())
    node_colors = [255 - (degree / max_degree * 255) for degree in node_degrees.values()]
    node_sizes = [5 + (degree / max_degree * 20) for degree in node_degrees.values()]

    # Define a custom colormap from light blue to red
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["lightgreen", "red"])
    norm = mcolors.Normalize(vmin=min(nx.get_edge_attributes(G, 'weight').values()),
                            vmax=max(nx.get_edge_attributes(G, 'weight').values()))

    # Create edge traces with custom colors
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        rgba_color = mcolors.to_rgba(custom_cmap(norm(weight)), alpha=0.5)
        color = f'rgba({rgba_color[0] * 255}, {rgba_color[1] * 255}, {rgba_color[2] * 255}, {rgba_color[3]})'

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=0.5, color=color),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # Create node traces
    node_x, node_y = zip(*[pos[node] for node in G.nodes()])
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=[f'Node: {node}<br>Degree: {degree}' for node, degree in node_degrees.items()],
        marker=dict(
            showscale=False,
            colorscale='Blues',
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            ),
            line_width=1,
            line_color='DarkSlateGrey'))

    # Combine edge and node traces
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='<b>Global Network Graph Edges</b>',
                        title_x=0.5,
                        titlefont=dict(size=24, color='DarkSlateGrey'),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[
                            dict(
                                showarrow=False,
                                text="Nodes are sized and colored based on their degree; edges colored by weight",
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(size=12)
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=1000,
                        height=1000
                    ))

    _, center, _ = st.columns([1, 10, 1])
    center.plotly_chart(fig)
