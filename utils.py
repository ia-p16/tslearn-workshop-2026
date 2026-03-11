from collections import Counter
import csv
import os.path

import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset


base_path = os.path.join('.', 'A multi-sensory dataset for the activities of daily living')


def get_X_y(
    volunteers=None,
    imus=None,
    features=None,
    task_labels=None,
    min_size=None, verbose=False
):
    # Get all volunteers / imus / features / tasks by default

    features_map = {
        "x-acc": 6,
        "y-acc": 7,
        "z-acc": 8,
        "x-vel": 9,
        "y-vel": 10,
        "z-vel": 11,
    }

    if volunteers is None:
        volunteers = list(range(1, 11))
    if imus is None:
        imus = ["back", "lla", "lua", "rla", "rt", "rua"]
    if features is None:
        features = ["x-acc", "y-acc", "z-acc", "x-vel", "y-vel", "y-vel"]
    features = [features_map[feature] for feature in features]

    dataset = []
    tasks = []

    for i in volunteers:

        vol_tasks = []

        # process annotation
        annotation_file = os.path.join(base_path, f"volunteer_{i:02}", "annotations.CSV")
        start, stop, task = None, None, None
        with open(annotation_file) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                label = row[2].lower().strip()
                if task_labels is not None and label not in task_labels:
                    continue
                if row[3].lower().strip() == "start" and label:
                    start = int(row[1])
                    task_label = label
                if row[3].lower().strip() == "end" and label == task_label:
                    stop = int(row[1])
                    vol_tasks.append([start, stop, task_label])
        tasks.extend(vol_tasks)

        vol_dataset = [{} for i in range(len(vol_tasks))]
        for imu in imus:
            data_file = os.path.join(base_path, f"volunteer_{i:02}", "IMUs", f"{imu}.csv")
            with open(data_file) as csvfile:
                reader = csv.reader(csvfile)
                task_iter = enumerate(vol_tasks)
                task_index, task = next(task_iter)
                record = False
                for row in reader:
                    if int(row[1]) >= task[0] and not record:
                        record = True
                    if int(row[1]) > task[1] and record:
                        record = False
                        try:
                            task_index, task = next(task_iter)
                        except StopIteration:
                            break
                    if record:
                        sample = [int(row[feature]) for feature in features]
                        vol_dataset[task_index].setdefault(row[1], {})[imu] = sample
        dataset.extend(vol_dataset)

    # Deal with missing data
    for task_index, task in enumerate(dataset):
        for timestamp, samples in task.items():
            for imu in imus:
                if imu not in samples:
                    if verbose:
                        print(f"task {task_index} {tasks[task_index]} with missing data", imu, timestamp)
                    samples[imu] = [np.nan] * len(features)

    dataset_final = []
    for task_index, task in enumerate(dataset):
        task_final = []
        for timestamp in sorted(task.keys()):
            data = task[timestamp]
            sample = []
            for imu in sorted(data.keys()):
                sample.extend(data[imu])
            task_final.append(sample)

        if min_size is None or min_size <= len(task_final):
            dataset_final.append(task_final)
        else:
            if verbose:
                print(f"task {task_index} {tasks[task_index]} of size {len(task_final)} skipped")
            tasks[task_index][2] = "to_remove"

    return to_time_series_dataset(dataset_final), [task for task in tasks if task[2] != "to_remove"]


# Visualize label counts, tasks timeline and raw features for the first n tasks.
def visualize_dataset_overview(labels, tasks, dataset, n_tasks_to_plot=10):

    counts = Counter(labels)

    unique_labels = np.unique(labels)
    task_colors = plt.get_cmap("Set3").colors
    color_map = {lbl: task_colors[i] for i, lbl in enumerate(unique_labels)}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2, 2]})

    # Bar plot of label counts
    ax1.bar(unique_labels, [counts.get(l, 0) for l in unique_labels], color=list(color_map.values()))
    ax1.set_title("Label counts")
    ax1.set_ylabel("Occurrences")

    # Timeline of the first n tasks
    tasks_to_plot = tasks[:n_tasks_to_plot]

    min_t = min(s for s, e, _ in tasks_to_plot) / 1000.0
    max_t = max(e for s, e, _ in tasks_to_plot) / 1000.0

    xticks = np.linspace(min_t, max_t, 10)
    xticks_int = np.ceil(xticks).astype(int)
    tick_labels = [str(int(t)) for t in xticks_int]

    y = np.arange(len(tasks_to_plot))
    for i, (s, e, l) in enumerate(tasks_to_plot):
        ax2.barh(i, e / 1000.0 - s / 1000.0, left=s / 1000.0, color=color_map[l])

    ax2.set_yticks(y)
    ax2.set_yticklabels([l for _, _, l in tasks_to_plot])
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"First {n_tasks_to_plot} Tasks timeline")
    ax2.set_ylim(-0.5, len(tasks_to_plot) - 0.5)
    ax2.set_xlim(min_t, max_t)
    ax2.set_xticks(xticks_int)
    ax2.set_xticklabels(tick_labels)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[l]) for l in unique_labels]
    ax2.legend(handles, unique_labels, loc='lower right', fontsize='small')

    # Raw features for the first n tasks
    feature_colors = plt.get_cmap("tab10").colors
    plotted_feat = set()
    for idx, (s, e, label) in enumerate(tasks_to_plot):
        s_start = s / 1000.0
        samples = dataset[idx, :, :3]
        timestamps = s_start + np.arange(samples.shape[0]) * 0.03
        for feat_idx in range(3):
            label_text = f"feature {feat_idx}" if feat_idx not in plotted_feat else None
            ax3.plot(timestamps, samples[:, feat_idx], color=feature_colors[feat_idx], alpha=0.5, label=label_text)
            plotted_feat.add(feat_idx)

    ax3.set_xlim(min_t, max_t)
    ax3.set_xticks(xticks_int)
    ax3.set_xticklabels(tick_labels)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Feature value")
    ax3.set_title(f"First three features (accelerations along the x, y and z axes of back IMU) for the first {n_tasks_to_plot} tasks")
    ax3.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.show()


def visualize_clustering_results(model, dataset, labels):
    cluster_labels = model.labels_
    unique_labels = np.unique(labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Build contingency table (true label vs cluster)
    le = LabelEncoder()
    true_encoded = le.fit_transform(labels)
    n_clusters = 3
    contingency = np.zeros((len(unique_labels), n_clusters), dtype=int)
    for true, pred in zip(true_encoded, cluster_labels):
        contingency[true, pred] += 1

    # Left: Contingency heatmap
    im = ax1.imshow(contingency, cmap="Blues")
    ax1.set_xticks(range(n_clusters))
    ax1.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
    ax1.set_yticks(range(len(unique_labels)))
    ax1.set_yticklabels(unique_labels)
    ax1.set_xlabel("Predicted cluster")
    ax1.set_ylabel("True label")
    ax1.set_title("True label vs. Cluster (contingency)")
    for i in range(len(unique_labels)):
        for j in range(n_clusters):
            ax1.text(j, i, contingency[i, j], ha="center", va="center",
                 color="white" if contingency[i, j] > contingency.max() / 2 else "black", fontsize=13)
    plt.colorbar(im, ax=ax1)

    # PCA for 2D projection
    _resampled = TimeSeriesResampler().fit_transform(dataset)
    n_samples, n_timesteps, n_feat = _resampled.shape
    flat = np.nan_to_num(_resampled.reshape(n_samples, n_timesteps * n_feat))
    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)

    cluster_colors = ["tab:blue", "tab:orange", "tab:green"]
    marker_map = {lbl: m for lbl, m in zip(unique_labels, ["o", "s", "^"])}

    # Right: PCA scatter plot colored by cluster and shaped by true label
    for ci in range(n_clusters):
        mask = cluster_labels == ci
        for lbl in unique_labels:
            lbl_mask = labels == lbl
            combined = mask & lbl_mask
            ax2.scatter(coords[combined, 0], coords[combined, 1],
                   color=cluster_colors[ci],
                   marker=marker_map[lbl],
                   label=f"Cluster {ci} / {lbl}" if combined.any() else "",
                   edgecolors="k", linewidths=0.5, s=70, alpha=0.8)

    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("PCA projection — color=cluster, shape=true label")
    handles, lbls = ax2.get_legend_handles_labels()
    ax2.legend([h for h, l in zip(handles, lbls) if l], [l for l in lbls if l],
          fontsize="small", loc="best")

    plt.tight_layout()
    plt.show()


def visualize_classification_results(classifier, test_labels, pred_labels):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    unique_labels = np.unique(test_labels)
    cm = confusion_matrix(test_labels, pred_labels, labels=unique_labels)

    # Confusion matrix heatmap (left)
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ticks = np.arange(len(unique_labels))
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels(unique_labels, rotation=45, ha='right')
    ax1.set_yticklabels(unique_labels)
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")
    ax1.set_title("Confusion Matrix")

    # Annotate cells
    fmt = 'd'
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(int(cm[i, j]), fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Classification report (right) as monospace text
    report = classification_report(test_labels, pred_labels,
                               labels=unique_labels, target_names=unique_labels,
                               zero_division=0)

    ax2.axis("off")
    ax2.set_title("Classification Report")
    ax2.text(0, 1, report, fontsize=12, fontfamily="monospace", transform=ax2.transAxes, va="top")

    plt.tight_layout()
    plt.show()


def visualize_shapelets(shapelet_clf, feature_names=None):
    n_shapelets = len(shapelet_clf.shapelets_)
    fig, axes = plt.subplots(1, n_shapelets, figsize=(5 * n_shapelets, 3))

    # handle single shapelet case
    if n_shapelets == 1:
        axes = [axes]

    feature_colors = plt.get_cmap("tab10").colors

    for i, shapelet in enumerate(shapelet_clf.shapelets_):
        # shapelet shape: (length, n_features)
        n_feat = shapelet.shape[1]
        for f in range(n_feat):
            label = feature_names[f] if feature_names else f"feature {f}"
            axes[i].plot(shapelet[:, f], color=feature_colors[f % len(feature_colors)],
                         label=label, linewidth=1.8)
        axes[i].set_title(f"Shapelet {i + 1}\n(len={len(shapelet)})")
        axes[i].legend(fontsize="small")
        axes[i].set_xlabel("Time step")

    plt.suptitle("Learned shapelets — all features", fontweight="bold")
    plt.tight_layout()
    plt.show()


def visualize_shapelet_matches(shapelet_clf, data, labels, show_feature=0):
    unique_lbls = np.unique(labels)
    n_shapelets = len(shapelet_clf.shapelets_)
    locations = shapelet_clf.locate(data)

    fig, axes = plt.subplots(n_shapelets, len(unique_lbls), figsize=(12, 3 * n_shapelets))

    # Ensure axes is always 2D
    if n_shapelets == 1:
        axes = axes[np.newaxis, :]
    if len(unique_lbls) == 1:
        axes = axes[:, np.newaxis]

    for s_idx in range(n_shapelets):
        shapelet = shapelet_clf.shapelets_[s_idx]
        sz = len(shapelet)

        for col, lbl in enumerate(unique_lbls):
            ax = axes[s_idx, col]
            # pick first series of this class
            idx = np.where(labels == lbl)[0][0]
            series = data[idx, :, show_feature]
            best_pos = locations[idx, s_idx]
            best_dist = np.linalg.norm(series[best_pos:best_pos + sz].reshape(-1) - shapelet.reshape(-1))

            feature_colors = plt.get_cmap("tab10").colors
            ax.plot(series, color=feature_colors[show_feature], alpha=0.5, label="feature series")

            ax.plot(range(best_pos, best_pos + sz), series[best_pos:best_pos + sz],
                    color="red", linewidth=2, label="best match")

            ax.plot(range(best_pos, best_pos + sz), shapelet[:, show_feature], color="blue",
                    linewidth=2, linestyle="--", label="shapelet")

            ax.set_title(f"Shapelet {s_idx + 1} | Class: {lbl} {idx}\ndist={best_dist:.2f}")
            ax.legend(fontsize="small")

    plt.suptitle(f"Shapelets vs. Class for feature {show_feature}", fontweight="bold")
    plt.tight_layout()
    plt.show()


# Plot cluster centroids (one plot per cluster, all features in each plot)
def visualize_cluster_centroids(model, feature_labels, time_index=None):

    centroids = model.cluster_centers_
    n_clusters, n_timesteps, n_features = centroids.shape

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 2.8 * max(1, n_clusters)), sharex=True)
    axes = [axes] if n_clusters == 1 else axes

    if time_index is None:
        time_index = np.arange(n_timesteps)
    else:
        time_index = np.asarray(list(time_index))
        if time_index.shape[0] != n_timesteps:
            time_index = np.arange(n_timesteps)

    n_plot_features = min(len(feature_labels), n_features)
    feature_colors = plt.get_cmap("tab10").colors[:n_plot_features]

    for k in range(n_clusters):
        ax = axes[k]

        # determine sensible y-limits using finite centroid values for this cluster
        vals = np.hstack([centroids[k, :, f] for f in range(n_plot_features)])
        finite = np.isfinite(vals)
        if finite.any():
            vmin, vmax = vals[finite].min(), vals[finite].max()
            pad = (vmax - vmin) * 0.08 if vmax > vmin else 0.1
            ax.set_ylim(vmin - pad, vmax + pad)

        for f in range(n_plot_features):
            ax.plot(time_index, centroids[k, :, f],
                    color=feature_colors[f],
                    label=feature_labels[f],
                    linewidth=1.8, alpha=0.9)

        ax.set_title(f"Cluster {k} centroid — all features")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize="small")

        # xticks: up to 10 nicely spaced ticks
        n_ticks = min(10, n_timesteps)
        tick_pos = np.linspace(0, n_timesteps - 1, n_ticks).astype(int)
        ax.set_xticks(time_index[tick_pos])
        ax.set_xticklabels([str(time_index[i]) for i in tick_pos], rotation=0)

    axes[-1].set_xlabel("Time index")
    fig.suptitle("Cluster centroids — per cluster (all features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
