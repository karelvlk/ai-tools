import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from utils.color_palette import get_color


def tsne_visualizer(data_dict, perplexity=30, learning_rate=200, max_iter=1000):
    # Ensure data_dict is a dictionary
    if not isinstance(data_dict, dict):
        raise ValueError(
            "Input data should be a dictionary with categories as keys and feature arrays as values."
        )

    # Standardize the data
    all_data = []
    labels = []
    categories = list(data_dict.keys())
    num_categories = len(categories)

    for category, features in data_dict.items():
        all_data.extend(features)
        labels.extend([category] * len(features))

    all_data = np.array(all_data)
    labels = np.array(labels)

    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data)

    # t-SNE for 2D and 3D
    tsne_1d = TSNE(
        n_components=1,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42,
    )
    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42,
    )
    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42,
    )

    transformed_1d = tsne_1d.fit_transform(all_data_scaled)
    transformed_2d = tsne_2d.fit_transform(all_data_scaled)
    transformed_3d = tsne_3d.fit_transform(all_data_scaled)

    # Plotting
    fig = plt.figure(figsize=(18, 5))

    # 1D Plot
    ax1 = fig.add_subplot(131)
    for i, category in enumerate(categories):
        indices = labels == category
        ax1.scatter(
            transformed_1d[indices],
            np.zeros_like(transformed_1d[indices]),
            label=category,
            color=get_color(i, num_categories),
        )
    ax1.set_title("t-SNE - 1D")
    ax1.set_xlabel("Principal Component 1")
    ax1.legend()

    # 2D Plot
    ax1 = fig.add_subplot(132)
    for i, category in enumerate(categories):
        indices = labels == category
        ax1.scatter(
            transformed_2d[indices, 0],
            transformed_2d[indices, 1],
            label=category,
            color=get_color(i, num_categories),
        )
    ax1.set_title("t-SNE - 2D")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.legend()

    # 3D Plot
    ax2 = fig.add_subplot(133, projection="3d")
    for i, category in enumerate(categories):
        indices = labels == category
        ax2.scatter(
            transformed_3d[indices, 0],
            transformed_3d[indices, 1],
            transformed_3d[indices, 2],
            label=category,
            color=get_color(i, num_categories),
        )
    ax2.set_title("t-SNE - 3D")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    ax2.set_zlabel("Component 3")
    ax2.legend()

    plt.show()

    # Interactive 3D Plot with Plotly
    fig_3d = go.Figure()
    for i, category in enumerate(categories):
        indices = labels == category
        fig_3d.add_trace(
            go.Scatter3d(
                x=transformed_3d[indices, 0],
                y=transformed_3d[indices, 1],
                z=transformed_3d[indices, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=get_color(i, num_categories),
                ),
                name=category,
            )
        )

    fig_3d.update_layout(
        title="t-SNE - 3D",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
    )

    fig_3d.show()
