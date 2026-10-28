import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

from .data_pipeline import FEATURES
from .membership import OUT
from .rules import OUT_RU

sns.set_theme(style="whitegrid")


def _save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def eda_plots(df, folder):
    cols = FEATURES + ["expert_risk"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(c)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    _save(fig, folder, "eda_hist.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Корреляции")
    _save(fig, folder, "eda_corr.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    sns.boxplot(data=z, ax=ax)
    ax.set_title("Boxplot стандартизованных показателей")
    plt.setp(ax.get_xticklabels(), rotation=25)
    _save(fig, folder, "eda_boxplot.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, c in zip(axes, ["downtime_pct", "defect_rate", "overdue_orders"]):
        sns.scatterplot(data=df, x=c, y="expert_risk", s=12, alpha=0.5, ax=ax)
        sns.regplot(data=df, x=c, y="expert_risk", scatter=False, color="red", ax=ax)
    _save(fig, folder, "eda_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="workshop", y="expert_risk", ax=ax)
    ax.set_title("Средний экспертный риск по цехам")
    _save(fig, folder, "eda_workshop.png")


def mf_plots(mf, folder, name="membership.png"):
    names = FEATURES + [OUT]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, var in zip(axes.ravel(), names):
        u = mf.universe(var)
        for t in mf.terms(var):
            ax.plot(u, mf.degree(var, t, u), label=t)
        ax.set_title(var)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    return _save(fig, folder, name)


def rules_graph(rules, folder):
    g = nx.DiGraph()
    in_nodes = sorted({f"{v}:{t}" for r in rules for v, t in r.inputs_used()})
    out_nodes = ["out:" + t for t in OUT_RU]
    pos = {}
    for i, n in enumerate(in_nodes):
        pos[n] = (0, -i * 1.0)
    for i, r in enumerate(rules):
        pos[r.id] = (1.6, -i * (len(in_nodes) / len(rules)))
    for i, n in enumerate(out_nodes):
        pos[n] = (3.2, -(i + 1) * len(in_nodes) / 5)
    colors = {"out:low": "#7fc97f", "out:moderate": "#ffd966", "out:high": "#fdae61", "out:critical": "#e06666"}
    for r in rules:
        for v, t in r.inputs_used():
            g.add_edge(f"{v}:{t}", r.id)
        g.add_edge(r.id, "out:" + r.out)
    fig, ax = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_nodes(g, pos, nodelist=in_nodes, node_color="#9ecae1", node_size=900, node_shape="s", ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=[r.id for r in rules], node_color="#d9d9d9", node_size=500, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=out_nodes, node_color=[colors[n] for n in out_nodes],
                           node_size=1800, node_shape="s", ax=ax)
    nx.draw_networkx_edges(g, pos, alpha=0.4, arrows=True, arrowsize=8, ax=ax)
    labels = {n: n.replace(":", "\n") for n in in_nodes + out_nodes}
    labels.update({r.id: r.id for r in rules})
    nx.draw_networkx_labels(g, pos, labels, font_size=6, ax=ax)
    ax.set_title("База правил: входные термы -> правила -> выходные категории")
    ax.axis("off")
    return _save(fig, folder, "rules_graph.png")


def compare_plot(mamdani_vals, sugeno_vals, folder, scenarios=None):
    fig, axes = plt.subplots(1, 3 if scenarios else 2, figsize=(16 if scenarios else 11, 4.8))
    axes[0].scatter(mamdani_vals, sugeno_vals, s=8, alpha=0.5)
    axes[0].plot([0, 100], [0, 100], "r--")
    axes[0].set_xlabel("Мамдани")
    axes[0].set_ylabel("Сугено")
    axes[0].set_title("Мамдани vs Сугено")
    sns.histplot(np.abs(np.asarray(mamdani_vals) - np.asarray(sugeno_vals)), kde=True, ax=axes[1])
    axes[1].set_title("Абсолютное расхождение")
    if scenarios:
        x = np.arange(len(scenarios))
        axes[2].bar(x - 0.2, [s["mamdani"] for s in scenarios.values()], 0.4, label="Мамдани")
        axes[2].bar(x + 0.2, [s["sugeno"] for s in scenarios.values()], 0.4, label="Сугено")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(list(scenarios))
        axes[2].legend()
        axes[2].set_title("Контрольные сценарии")
    return _save(fig, folder, "compare_mamdani_sugeno.png")


def cluster_plots(table, cntr, scaler, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(table["c"], table["FPC"], "o-")
    axes[0].set_title("FPC")
    axes[1].plot(table["c"], table["Silhouette"], "o-", color="green")
    axes[1].set_title("Silhouette")
    for ax in axes:
        ax.set_xlabel("число кластеров")
    _save(fig, folder, "fcm_metrics.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(cntr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=FEATURES, ax=ax)
    ax.set_title("Центры кластеров (стандартизованные)")
    plt.setp(ax.get_xticklabels(), rotation=25)
    _save(fig, folder, "fcm_centers.png")


def optimization_plot(y, before, after, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, p, t in zip(axes, (before, after), ("до оптимизации", "после оптимизации")):
        ax.scatter(y, p, s=8, alpha=0.5)
        ax.plot([0, 100], [0, 100], "r--")
        ax.set_xlabel("expert_risk")
        ax.set_ylabel("Мамдани")
        ax.set_title(t + " (тест)")
    _save(fig, folder, "optimization.png")
