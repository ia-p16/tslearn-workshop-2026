from collections import Counter
import csv
import os.path

import matplotlib.pyplot as plt

import numpy as np

from tslearn.utils import to_time_series_dataset

# Rename volunteer_01/annotation.CSV -> volunteer_01/annotations.CSV
base_path = os.path.join('.', 'A multi-sensory dataset for the activities of daily living')


def get_X_y(volunteers=None, imus=None, features=None, task_labels=None, min_size=None):
    # Get all volunteers / imus / features / tasks by default

    features_map = {
        "x-acc": 6,
        "y-acc": 7,
        "z-acc": 8,
        "x-vel": 9,
        "y-vel": 10,
        "y-vel": 11,
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
                    # print(f"task {task_index} {tasks[task_index]} with missing data", imu, timestamp)
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
