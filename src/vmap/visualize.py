"""
Viralmap
Version: v1.0
ADAPT (2025)
"""

# // imports
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# // visualize
class VMAPViz():
    """
    ViralMap visualization
    """
    def __init__(self):
        vmap_blue = "#183359"
        self.tracks_color_map = {
            "SP": vmap_blue, "TM": vmap_blue, "CY": vmap_blue, "EX": vmap_blue,
            "NG": vmap_blue, "FR": vmap_blue, "CH": vmap_blue, "DB": vmap_blue,
            "CC": vmap_blue, "DR": vmap_blue,
        }
        self.category_layout = {
            "Topology & Localization": [
                ("SP", "Signal Peptide"),
                ("TM", "Transmembrane"),
                ("CY", "Cytoplasmic"),
                ("EX", "Extracellular"),
            ],
            "Post-translational Modifications": [
                ("NG", "N-Glycosylation"),
                ("FR", "Furin Cleavage"),
                ("CH", "Chain Cleavage"),
                ("DB", "Disulfide Bond"),
            ],
            "Structural Features": [
                ("CC", "Coiled Coil"),
                ("DR", "Disordered Region"),
            ],
        }
    
    # //
    def build_summary(self, df: pd.DataFrame):
        """
        build a per-feature summary table of predicted regions/positions
        """
        rows = []
        all_features = [
            ("SP", "Signal Peptide"), ("TM", "Transmembrane"),
            ("CY", "Cytoplasmic"), ("EX", "Extracellular"),
            ("NG", "N-Glycosylation"), ("FR", "Furin Cleavage"),
            ("CH", "Chain Cleavage"), ("DB", "Disulfide Bond Site"),
            ("CC", "Coiled Coil"), ("DR", "Disordered Region")]
        point_features = {"NG", "FR", "CH", "DB"}

        for feat_code, feat_name in all_features:
            if feat_code not in df.columns:
                continue
            binary = df[feat_code].to_numpy()

            runs = []
            in_run = False
            start = None
            for idx, v in enumerate(binary):
                if v == 1 and not in_run:
                    start = idx + 1
                    in_run = True
                elif v == 0 and in_run:
                    runs.append((start, idx))
                    in_run = False
            if in_run:
                runs.append((start, len(binary)))

            if feat_code in point_features:
                # // 
                detail_parts = []
                for s, e in runs:
                    if s == e:
                        detail_parts.append(str(s))
                    else:
                        detail_parts.append(f"[{s},{e}]")
                detail = ", ".join(detail_parts) if runs else "n/a"
                
            else:
                detail = ", ".join(f"[{s},{e}]" if s != e else str(s) for s, e in runs) if runs else "n/a"

            rows.append({
                "Feature": feat_name,
                "Regions / Positions": detail,
                "Count": len(runs)})

        return pd.DataFrame(rows)


    def render_interactive(self, df: pd.DataFrame, save_out: str | None = None, name: str = "protein", title: str = None):
        """
        interactive output
        """

        L = len(df)
        residues = np.arange(1, L + 1)
        blue = "#183359"

        topo_features   = [("SP", "Signal Peptide"), ("TM", "Transmembrane"),("CY", "Cytoplasmic"), ("EX", "Extracellular")]
        ptm_features    = [("NG", "N-Glycosylation"), ("FR", "Furin Cleavage"),("CH", "Chain Cleavage"), ("DB", "Disulfide Bond")]
        struct_features = [("CC", "Coiled Coil"),     ("DR", "Disordered Region")]

        all_feats = topo_features + ptm_features + struct_features
        point_features = {"NG", "FR", "CH", "DB"}
        n_plot_rows = len(all_feats)

        aa_seq = df["AA"].to_numpy() if "AA" in df.columns else None

        row_heights = [0.05] * n_plot_rows + [0.55]
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]

        row_labels = [label for _, label in all_feats]
        specs = [[{"type": "xy"}]] * n_plot_rows + [[{"type": "table"}]]

        fig = make_subplots(
            rows=n_plot_rows + 1, cols=1,
            specs=specs,
            shared_xaxes=False,
            vertical_spacing=0.035,
            row_heights=row_heights)

        # // tracks
        for idx, (feat, label) in enumerate(all_feats, start=1):
            if feat not in df.columns:
                continue
            binary = df[feat].to_numpy()
            probs = df[f"{feat}_prob"].to_numpy() if f"{feat}_prob" in df.columns else None

            if feat in point_features:
                positions = residues[binary == 1]
                if len(positions) > 0:
                    hover_texts = [
                        f"<b>{label}</b><br>Residue {p}" +
                        (f" ({aa_seq[p-1]})" if aa_seq is not None else "") +
                        (f"<br>Probability: {probs[p-1]:.3f}" if probs is not None else "")
                        for p in positions]
                    fig.add_trace(
                        go.Scatter(
                            x=positions, y=[0.5] * len(positions),
                            mode='markers',
                            marker=dict(symbol='line-ns', size=22,
                                        color=blue, line=dict(width=2)),
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_texts,
                            showlegend=False),row=idx, col=1)
            else:
                # // contiguous runs as shape rectangles 
                runs = []
                in_run, start = False, None
                for i, v in enumerate(binary):
                    if v == 1 and not in_run:
                        start, in_run = i + 1, True
                    elif v == 0 and in_run:
                        runs.append((start, i + 1 - start))
                        in_run = False
                if in_run:
                    runs.append((start, len(binary) + 1 - start))

                for s, w in runs:
                    region_prob = probs[s-1:s-1+w].mean() if probs is not None else None
                    fig.add_shape(
                        type="rect",
                        x0=s, x1=s + w, y0=0.2, y1=0.8,
                        fillcolor=blue, line=dict(width=0),
                        row=idx, col=1,
                    )
                    # // invisible hover anchor for the region
                    hover = (f"<b>{label}</b><br>Residues {s}–{s + w - 1}" +
                            (f"<br>Mean probability: {region_prob:.3f}" if region_prob is not None else ""))
                    fig.add_trace(
                        go.Scatter(
                            x=[s + w/2], y=[0.5],
                            mode='markers',
                            marker=dict(size=1, color='rgba(0,0,0,0)'),
                            hovertemplate=hover + "<extra></extra>",
                            showlegend=False,
                        ),
                        row=idx, col=1,
                    )

                # // per-residue hover
                if probs is not None:
                    if aa_seq is not None:
                        hover_text = [f"{aa_seq[i]} — prob: {probs[i]:.3f}" for i in range(L)]
                        hovertemplate = f"<b>{label}</b><br>Residue %{{x}}<br>%{{text}}<extra></extra>"
                    else:
                        hover_text = [f"prob: {p:.3f}" for p in probs]
                        hovertemplate = f"<b>{label}</b><br>Residue %{{x}}<br>%{{text}}<extra></extra>"

                    fig.add_trace(
                        go.Scatter(
                            x=residues, y=[0.5] * L,
                            mode='markers',
                            marker=dict(size=8, color='rgba(0,0,0,0)'),
                            hovertemplate=hovertemplate,
                            text=hover_text,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white"),
                        ),
                        row=idx, col=1)

        # // summary table
        summary = self.build_summary(df)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in summary.columns],
                    fill_color=blue,
                    font=dict(color="white", family="Arial", size=11),
                    align="left", height=28,
                ),
                cells=dict(
                    values=[summary[c] for c in summary.columns],
                    fill_color=[["#F8FAFC", "#FFFFFF"] * len(summary)],
                    font=dict(family="Arial", size=10, color="#1E293B"),
                    align="left", height=24,
                ),
            ),
            row=n_plot_rows + 1, col=1,
        )

        # // axes
        for r in range(1, n_plot_rows + 1):
            fig.update_yaxes(
                tickmode='array', tickvals=[0.5], ticktext=[row_labels[r-1]],
                showticklabels=True,
                showgrid=False, zeroline=False,
                range=[0, 1],
                ticklabelstandoff=15,
                row=r, col=1,
            )
            fig.update_xaxes(
                showgrid=False, zeroline=False,
                range=[1, L],
                showticklabels=(r == n_plot_rows),
                showline=True, linewidth=1, linecolor='#E2E8F0',
                row=r, col=1,
            )
        fig.update_xaxes(
            tickmode='linear', tick0=0, dtick=max(100, L // 10),
            row=n_plot_rows, col=1)

        # // layout
        width, height = 1200, 1000
        l_margin, r_margin = 200, 60
        plot_center_x = (l_margin + (width - l_margin - r_margin) / 2) / width

        # // labels
        fig.add_annotation(
            text="N-terminus", xref="x1", yref="paper",
            x=1, y=1.02, xanchor='left',
            showarrow=False,
            font=dict(family="Arial", size=10, color='gray'))
        fig.add_annotation(
            text="C-terminus", xref="x1", yref="paper",
            x=L, y=1.02, xanchor='right',
            showarrow=False,
            font=dict(family="Arial", size=10, color='gray'))
  
        fig.update_layout(
            title=dict(
                text=f"{title or name}<br><span style='font-size:11px;color:#64748B;font-weight:normal'>Length: {L} AA</span>",
                font=dict(size=15, family="Arial", color="#0F1F3D"),
                x=plot_center_x, xanchor='center',
                y=0.98, yanchor='top',
            ),
            height=1200, width=width,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=l_margin, r=r_margin, t=80, b=80),
            hovermode='closest',
            showlegend=False,
            font=dict(family="Arial"))

        if save_out:
            fig_path = os.path.join(save_out, f"{name}_viz.html")
            fig.write_html(fig_path, include_plotlyjs='cdn', config={'displayModeBar': False})

        return fig, summary