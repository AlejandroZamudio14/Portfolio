from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

import ast
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from mplsoccer import Pitch

import plotly.graph_objects as go

import base64
from io import BytesIO


schedule = pd.read_csv("Data/2022WorldCup_Schedule.csv")
team_stats = pd.read_csv("Data/TeamStats_WC2022.csv")
team_cols = pd.read_csv("Data/soccer_national_team_primary_colors.csv")
starters_df = pd.read_csv("Data/2022WorldCup_Starters.csv", encoding="utf-8-sig")
sub_entries = pd.read_csv("Data/2022WorldCup_SubEntry.csv")
rosters_df = pd.read_csv("Data/2022WorldCup_FullRosters.csv")
player_mapping_df = pd.read_csv("Data/PlayerMapping_WC2022.csv")
standard_stats = pd.read_csv("Data/2022WorldCup_StandardData.csv")
attacker_stats = pd.read_csv("Data/2022WorldCup_AttackerData.csv")
midfielder_stats = pd.read_csv("Data/2022WorldCup_MidfielderData.csv")
defender_stats = pd.read_csv("Data/2022WorldCup_DefenderData.csv")
keeper_stats = pd.read_csv("Data/2022WorldCup_KeeperData.csv")
visual_data = pd.read_csv("Data/2022WorldCup_VisualizationData.csv")
penalty_data = pd.read_csv("Data/2022WorldCup_PenaltyData.csv")
similar_performances_df = pd.read_csv("Data/2022WorldCup_KNN_Players.csv")


xT_grid = pd.read_csv("Data/xT_Grid_Values.csv", header=None, skiprows=1).to_numpy()
xT_rows, xT_cols = xT_grid.shape


def parse_location(value):
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None

    return value


location_cols = [
    "location",
    "pass_end_location",
    "carry_end_location"
]

for col in location_cols:
    visual_data[col] = visual_data[col].apply(parse_location)

def fig_to_data_uri(fig):
    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=150
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"

team_colors = dict(
    zip(
        team_cols["Country"],
        team_cols["primary_color_hex"]
    )
)

# X coordinates: depth of each formation line
xCords = {
    "3": [25, 40, 55],
    "4": [22, 33, 44, 55],
    "5": [22, 30, 38, 46, 55]
}

# Y coordinates: vertical placement of players within each line
yCords = {
    "1": [40],
    "2": [55, 25],
    "3": [60, 40, 20],
    "4": [70, 50, 30, 10],
    "5": [70, 55, 40, 25, 10]
}

# StatsBomb position ID -> vertical ordering within a line
pitch_order = {
    1: 1,   # Goalkeeper

    2: 2,   # Right Back
    3: 3,   # Right Center Back
    4: 4,   # Center Back
    5: 5,   # Left Center Back
    6: 6,   # Left Back

    7: 1,   # Right Wing Back
    8: 7,   # Left Wing Back

    9: 2,   # Right Defensive Midfield
    10: 4,  # Center Defensive Midfield
    11: 6,  # Left Defensive Midfield

    12: 2,  # Right Midfield
    13: 3,  # Right Center Midfield
    14: 4,  # Center Midfield
    15: 5,  # Left Center Midfield
    16: 6,  # Left Midfield

    17: 1,  # Right Wing
    18: 2,  # Right Attacking Midfield
    19: 4,  # Center Attacking Midfield
    20: 6,  # Left Attacking Midfield
    21: 5,  # Left Wing

    22: 2,  # Right Center Forward
    23: 3,  # Center Forward
    24: 4,  # Left Center Forward
    25: 3   # Secondary Striker
}

team_finishes = {
    # Champions and medal positions
    "Argentina": "Champions",
    "France": "Runner-up",
    "Croatia": "3rd Place",
    "Morocco": "4th Place",

    # Quarter-final exits
    "Netherlands": "Quarter-finals",
    "Brazil": "Quarter-finals",
    "England": "Quarter-finals",
    "Portugal": "Quarter-finals",

    # Round of 16 exits
    "United States": "Round of 16",
    "Senegal": "Round of 16",
    "Australia": "Round of 16",
    "Poland": "Round of 16",
    "Japan": "Round of 16",
    "Spain": "Round of 16",
    "Switzerland": "Round of 16",
    "South Korea": "Round of 16",

    # Group-stage exits
    "Qatar": "Group Stage",
    "Ecuador": "Group Stage",
    "Wales": "Group Stage",
    "Iran": "Group Stage",
    "Mexico": "Group Stage",
    "Saudi Arabia": "Group Stage",
    "Tunisia": "Group Stage",
    "Denmark": "Group Stage",
    "Germany": "Group Stage",
    "Costa Rica": "Group Stage",
    "Belgium": "Group Stage",
    "Canada": "Group Stage",
    "Serbia": "Group Stage",
    "Cameroon": "Group Stage",
    "Uruguay": "Group Stage",
    "Ghana": "Group Stage"
}

def calculate_xt(events):
    events = events[
        events["type"].isin(["Pass", "Carry"])
    ].copy()

    # Choose the appropriate ending location for each event
    events["end_location"] = [
        carry_end if event_type == "Carry" else pass_end
        for event_type, pass_end, carry_end in zip(
            events["type"],
            events["pass_end_location"],
            events["carry_end_location"]
        )
    ]

    # Remove events without usable start or end coordinates
    events = events.dropna(
        subset=["location", "end_location"]
    ).copy()

    events = events[
        events["location"].apply(
            lambda loc: isinstance(loc, (list, tuple)) and len(loc) >= 2
        )
        & events["end_location"].apply(
            lambda loc: isinstance(loc, (list, tuple)) and len(loc) >= 2
        )
    ].copy()

    # Extract x and y coordinates
    events["x"] = events["location"].apply(lambda loc: loc[0])
    events["y"] = events["location"].apply(lambda loc: loc[1])
    events["end_x"] = events["end_location"].apply(lambda loc: loc[0])
    events["end_y"] = events["end_location"].apply(lambda loc: loc[1])

    # Assign starting and ending locations to grid cells
    events["start_x_bin"] = np.floor(
        events["x"] / 120 * xT_cols
    ).clip(0, xT_cols - 1).astype(int)

    events["start_y_bin"] = np.floor(
        events["y"] / 80 * xT_rows
    ).clip(0, xT_rows - 1).astype(int)

    events["end_x_bin"] = np.floor(
        events["end_x"] / 120 * xT_cols
    ).clip(0, xT_cols - 1).astype(int)

    events["end_y_bin"] = np.floor(
        events["end_y"] / 80 * xT_rows
    ).clip(0, xT_rows - 1).astype(int)

    # Look up the xT value in each grid cell
    events["start_zone_value"] = xT_grid[
        events["start_y_bin"], events["start_x_bin"]
    ]

    events["end_zone_value"] = xT_grid[
        events["end_y_bin"], events["end_x_bin"]
    ]

    # Change in expected threat
    events["xT"] = (
        events["end_zone_value"] - events["start_zone_value"]
    )

    return events[
        ["match_id", "minute", "second", "team", "type", "xT"]
    ]

def calculate_momentum(events, home_team, away_team, window_size=4, decay_rate=0.25):
    xt_events = calculate_xt(events).copy()

    # Keep positive xT contributions, matching the original
    # momentum approach we are adapting.
    xt_events["positive_xT"] = xt_events["xT"].clip(0, 0.1)

    # Find each team's maximum xT contribution per minute.
    minute_xt = (
        xt_events.groupby(["team", "minute"])["positive_xT"]
        .max()
        .reset_index()
    )

    # Include every minute so gaps in activity are represented.
    minutes = np.arange(
        int(events["minute"].min()),
        int(events["minute"].max()) + 1
    )

    momentum = []

    for minute in minutes:
        team_values = {}

        for team in [home_team, away_team]:
            recent = minute_xt[
                (minute_xt["team"] == team)
                & (minute_xt["minute"] <= minute)
                & (minute_xt["minute"] >= minute - window_size + 1)
            ]

            weighted_xt = (
                recent["positive_xT"]
                * np.exp(
                    -decay_rate * (minute - recent["minute"])
                )
            ).sum()

            team_values[team] = weighted_xt

        momentum.append({
            "minute": minute,
            "momentum": team_values[home_team] - team_values[away_team]
        })

    momentum_df = pd.DataFrame(momentum)

    momentum_df["smoothed_momentum"] = gaussian_filter1d(
        momentum_df["momentum"].to_numpy(),
        sigma=1
    )

    return momentum_df



def plot_xt_momentum(events, home_team, away_team):
    momentum_df = calculate_momentum(events, home_team, away_team)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.set_facecolor("White")
    ax.set_facecolor("White")

    ax.tick_params(axis="x", colors="Black")
    ax.tick_params(
        axis="y", which="both",
        left=False, right=False, labelleft=False
    )

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    max_minute = int(momentum_df["minute"].max())
    ax.set_xticks(range(0, max_minute + 1, 15))

    if max_minute > 90 and max_minute % 15 != 0:
        ax.set_xticks(
            list(ax.get_xticks()) + [max_minute]
        )
    ax.margins(x=0)
    momentum_limit = max(
        abs(momentum_df["smoothed_momentum"].min()),
        abs(momentum_df["smoothed_momentum"].max())
    )

    ax.set_ylim(-momentum_limit * 1.15, momentum_limit * 1.15)

    momentum_df["smoothed_momentum"] = gaussian_filter1d(
        momentum_df["momentum"], sigma=1
    )

    ax.plot(
        momentum_df["minute"],
        momentum_df["smoothed_momentum"],
        color="Black"
    )

    ax.axhline(0, color="Black", linestyle="--", linewidth=0.5)

    ax.fill_between(
        momentum_df["minute"],
        momentum_df["smoothed_momentum"],
        where=momentum_df["smoothed_momentum"] > 0,
        color=team_colors.get(home_team, "#0053a0"),
        alpha=0.5,
        interpolate=True
    )

    ax.fill_between(
        momentum_df["minute"],
        momentum_df["smoothed_momentum"],
        where=momentum_df["smoothed_momentum"] < 0,
        color=team_colors.get(away_team, "#DA291C"),
        alpha=0.5,
        interpolate=True
    )

    ax.set_xlabel(
        "Minute", color="Black", fontsize=15,
        fontweight="bold", fontfamily="Monospace"
    )
    ax.set_ylabel(
        "Momentum", color="Black", fontsize=15,
        fontweight="bold", fontfamily="Monospace"
    )
    ax.set_title(
        "xT Momentum", color="Black", fontsize=20,
        fontweight="bold", fontfamily="Monospace", pad=10
    )

    home_team_text = ax.text(
        7, momentum_limit * 0.72,
        home_team,
        fontsize=12,
        ha="center",
        fontfamily="Monospace",
        fontweight="bold",
        color="white"
    )
    home_team_text.set_bbox(dict(
        facecolor=team_colors.get(home_team, "#0053a0"),
        alpha=0.7, edgecolor="white", boxstyle="round"
    ))

    away_team_text = ax.text(
        7, -momentum_limit * 0.72,
        away_team,
        fontsize=12,
        ha="center",
        fontfamily="Monospace",
        fontweight="bold",
        color="white"
    )
    away_team_text.set_bbox(dict(
        facecolor=team_colors.get(away_team, "#DA291C"),
        alpha=0.7, edgecolor="white", boxstyle="round"
    ))

    # Exclude penalty-shootout period 5 from goal annotations
    goals = events[
        (events["type"] == "Shot")
        & (events["shot_outcome"] == "Goal")
        & (events["period"] != 5)
    ][["minute", "team"]]

    for _, row in goals.iterrows():
        is_home = row["team"] == home_team

        y_position = (
            momentum_limit * 0.85
            if is_home
            else -momentum_limit * 0.85
        )

        ymin, ymax = (0.5, 0.8) if is_home else (0.2, 0.5)

        ax.axvline(
            row["minute"],
            color="Black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            ymin=ymin,
            ymax=ymax
        )

        ax.scatter(
            row["minute"],
            y_position,
            color="Black",
            s=65,
            zorder=10,
            alpha=0.7
        )

        ax.annotate(
            "Goal",
            (row["minute"], y_position),
            xytext=(4, 3 if is_home else -10),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom" if is_home else "top",
            fontfamily="Monospace",
            color="Black"
        )
        
    fig.tight_layout(pad=2)

    return fig


def plot_cumulative_xg(events, home_team, away_team):
    events = events[events["period"] != 5].copy()

    shots = events[
        (events["type"] == "Shot")
    ].copy()

    shots["xG"] = pd.to_numeric(
        shots["shot_statsbomb_xg"], errors="coerce"
    ).fillna(0)

    max_minute = int(events["minute"].max())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    
    # One cumulative xG value per minute for each team.
    for team in [home_team, away_team]:
        team_shots = shots[shots["team"] == team].copy()

        # Sum xG from shots in each minute.
        xg_by_minute = team_shots.groupby("minute")["xG"].sum()

        # Create a complete minute-by-minute timeline.
        minute_range = range(0, max_minute + 1)

        cumulative_xg = (
            xg_by_minute.reindex(minute_range, fill_value=0)
            .cumsum()
        )

        color = team_colors.get(
            team, "#0053a0" if team == home_team else "#DA291C"
        )

        ax.step(
            minute_range,
            cumulative_xg,
            where="post",
            color=color,
            linewidth=2,
            label=team
        )


    ax.set_xlim(0, max_minute)
    ax.set_xticks(range(0, max_minute + 1, 15))
    ax.set_xlabel(
        "Minute", fontsize=12, fontweight="bold", fontfamily="Monospace"
    )
    ax.set_ylabel(
        "Cumulative xG", fontsize=12, fontweight="bold",
        fontfamily="Monospace"
    )
    ax.set_title(
        "Cumulative Expected Goals",
        fontsize=16, fontweight="bold", fontfamily="Monospace", pad=10
    )

    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.margins(y=0.15)
    fig.tight_layout(pad=2)

    return fig



def plot_shot_map(events, home_team, away_team):
    shots = events[
        (events["type"] == "Shot")
        & (events["period"] != 5)
    ].copy()

    shots["xG"] = pd.to_numeric(
        shots["shot_statsbomb_xg"], errors="coerce"
    )

    shots = shots.dropna(subset=["location", "xG"])

    shots = shots[
        shots["location"].apply(
            lambda loc: isinstance(loc, (list, tuple)) and len(loc) >= 2
        )
    ].copy()

    shots["x"] = shots["location"].apply(lambda loc: loc[0])
    shots["y"] = shots["location"].apply(lambda loc: loc[1])

    # Team colors
    home_color = "#DA291C"
    away_color = "#0053A0"

    # Mirror away shots so both teams attack toward the right.
    shots["plot_x"] = shots["x"]
    shots["plot_y"] = shots["y"]

    away_mask = shots["team"] == away_team
    shots.loc[away_mask, "plot_x"] = 120 - shots.loc[away_mask, "x"]
    shots.loc[away_mask, "plot_y"] = 80 - shots.loc[away_mask, "y"]

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="#444444",
        linewidth=1
    )

    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor("white")

    # Circle size increases with xG.
    shots["marker_size"] = 70 + shots["xG"].clip(0, 1) * 900

    for team, color in [
        (home_team, home_color),
        (away_team, away_color)
    ]:
        team_shots = shots[shots["team"] == team]

        pitch.scatter(
            team_shots["plot_x"],
            team_shots["plot_y"],
            s=team_shots["marker_size"],
            color=color,
            edgecolors="white",
            linewidths=1,
            alpha=0.8,
            ax=ax,
            zorder=3
        )

        # Overlay a star on goals so they stand out from other shots.
        goals = team_shots[
            team_shots["shot_outcome"] == "Goal"
        ]

        if not goals.empty:
            pitch.scatter(
                goals["plot_x"],
                goals["plot_y"],
                s=goals["marker_size"] * 0.65,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=1,
                ax=ax,
                zorder=5
            )

    # Legend: team colors, goal marker, and xG size scale.
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=home_color, markeredgecolor="white",
            markersize=9, label=home_team
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=away_color, markeredgecolor="white",
            markersize=9, label=away_team
        ),
        Line2D(
            [0], [0], marker="*", linestyle="None",
            markerfacecolor="white", markeredgecolor="black",
            markersize=11, label="Goal"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor="#888888", markeredgecolor="white",
            markersize=5, label="Lower xG"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor="#888888", markeredgecolor="white",
            markersize=12, label="Higher xG"
        )
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    ax.set_title(
        f"Shot Map: {home_team} vs {away_team}",
        fontsize=15,
        fontweight="bold",
        fontfamily="Monospace",
        pad=12
    )

    fig.tight_layout(pad=2)
    return fig

def plot_player_pass_map(events, team_color):
    # Keep only pass events.
    passes = events[
        events["type"] == "Pass"
    ].copy()

    # Remove missing locations.
    passes = passes.dropna(
        subset=["location", "pass_end_location"]
    )

    # Keep valid coordinate pairs.
    passes = passes[
        passes["location"].apply(
            lambda loc: isinstance(loc, (list, tuple, np.ndarray))
            and len(loc) >= 2
        )
        &
        passes["pass_end_location"].apply(
            lambda loc: isinstance(loc, (list, tuple, np.ndarray))
            and len(loc) >= 2
        )
    ].copy()

    if passes.empty:
        return None

    # Create pitch.
    pitch = Pitch(
        pitch_type="statsbomb",
        line_color="#444444",
        pitch_color="white"
    )

    fig, ax = pitch.draw(figsize=(10, 7))

    # Plot passes as thin, transparent arrows.
    for _, row in passes.iterrows():
        start_x, start_y = row["location"][:2]
        end_x, end_y = row["pass_end_location"][:2]

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="->",
                color=team_color,
                linewidth=0.8,
                alpha=0.4,
                mutation_scale=8
            ),
            zorder=2
        )

    return fig

def plot_player_heat_map(events):
    # Keep events with valid location coordinates
    events = events.dropna(subset=["location"]).copy()

    events = events[
        events["location"].apply(
            lambda loc: isinstance(loc, (list, tuple, np.ndarray))
            and len(loc) >= 2
        )
    ]

    if events.empty:
        return None

    # Extract x and y coordinates
    x = events["location"].apply(lambda loc: loc[0]).to_numpy(dtype=float)
    y = events["location"].apply(lambda loc: loc[1]).to_numpy(dtype=float)

    # Remove invalid coordinates
    valid = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= 0) & (x <= 120)
        & (y >= 0) & (y <= 80)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return None

    # Create a 2D event density grid
    heatmap, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=[40, 27],
        range=[[0, 120], [0, 80]]
    )

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=2)


    # Draw the pitch
    pitch = Pitch(
        pitch_type="statsbomb",
        line_color="#444444",
        pitch_color="white"
    )

    fig, ax = pitch.draw(figsize=(10, 7))

    # Overlay the smoothed heat map
    ax.imshow(
        heatmap.T,
        extent=[0, 120, 0, 80],
        origin="lower",
        cmap="Reds",
        alpha=0.7,
        aspect="auto",
        interpolation="bilinear"
    )

    # Attacking direction label
    ax.text(
        0.5, 1.12,
        "ATTACKING DIRECTION",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="black"
    )

    # Centered arrow pointing right
    ax.annotate(
        "",
        xy=(0.58, 1.06),
        xytext=(0.42, 1.06),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=2
        ),
        annotation_clip=False
    )

    return fig


def plot_player_shot_map(events):
    # Keep only shot events
    shots = events[events["type"] == "Shot"].copy()

    shots = shots.dropna(subset=["location"])

    shots = shots[
        shots["location"].apply(
            lambda loc: isinstance(loc, (list, tuple, np.ndarray))
            and len(loc) >= 2
        )
    ].copy()

    if shots.empty:
        return None

    # Extract shot coordinates
    x = shots["location"].apply(lambda loc: loc[0]).to_numpy(dtype=float)
    y = shots["location"].apply(lambda loc: loc[1]).to_numpy(dtype=float)

    # Extract xG and shot outcomes
    xg = pd.to_numeric(
        shots["shot_statsbomb_xg"], errors="coerce"
    ).fillna(0).to_numpy()

    outcomes = shots["shot_outcome"].fillna("Unknown")

    # Draw full pitch
    pitch = Pitch(
        pitch_type="statsbomb",
        line_color="#444444",
        pitch_color="white"
    )

    fig, ax = pitch.draw(figsize=(10, 7))

    # Shot outcome colors
    outcome_colors = {
        "Goal": "#2E8B57",
        "Saved": "#2878B5",
        "Blocked": "#E69F00",
        "Off T": "#888888",
        "Wayward": "#888888",
        "Post": "#9B59B6",
        "Saved to Post": "#2878B5",
        "Saved Off Target": "#2878B5",
        "Deflected": "#888888",
        "Unknown": "#888888"
    }

    # Plot shots individually
    for outcome, color in outcome_colors.items():
        mask = outcomes == outcome

        if not mask.any():
            continue

        ax.scatter(
            x[mask],
            y[mask],
            s=40 + (xg[mask] * 500),
            c=color,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.85,
            label=outcome,
            zorder=3
        )

    # Legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False,
        fontsize=9
    )

    return fig


def draw_goal(ax):
    post_color="black"
    x1=[44,44]
    y1=[0,2.67]
    ax.plot(x1,y1,c=post_color,linewidth=15,zorder=1)
        
    x2=[36,36]
    y2=[0,2.67]
    ax.plot(x2,y2,c=post_color,linewidth=15,zorder=1)
    
    x3=[44,36]
    y3=[2.67,2.67]
    ax.plot(x3,y3,c=post_color,linewidth=15,zorder=1)
    
    x4=[41.33,41.33]
    y4=[0,2.67]
    ax.plot(x4,y4,c=post_color,linewidth=2.5,zorder=1,linestyle="dashed")

    x5=[38.66,38.66]
    y5=[0,2.67]
    ax.plot(x5,y5,c=post_color,linewidth=2.5,zorder=1,linestyle="dashed")


def plot_penalty_visual(data, home_team, away_team, penalty_type):
    data = data.copy()

    # Parse shot-end coordinates: [x, y, z].
    data["end_location"] = data["shot_end_location"].apply(
        parse_location
    )

    data = data.dropna(subset=["end_location"]).copy()

    data = data[
        data["end_location"].apply(
            lambda loc: isinstance(loc, (list, tuple))
            and len(loc) >= 3
        )
    ].copy()

    if data.empty:
        return None

    data["end_y"] = data["end_location"].apply(lambda loc: loc[1])
    data["end_z"] = data["end_location"].apply(lambda loc: loc[2])

    # Order attempts separately for each team.
    data = data.sort_values("index").copy()
    data["order"] = data.groupby("team").cumcount() + 1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_goal(ax)

    # Distinguish outcomes by marker shape.
    outcome_markers = {
        "Goal": "o",
        "Saved": "X",
        "Saved to Post": "X",
        "Post": "D",
        "Wayward": "D",
        "Off T": "D",
        "Saved Off Target": "D",
        "Blocked": "D",
    }

    for team in [home_team, away_team]:
        team_data = data[data["team"] == team]
        color = team_colors.get(
            team,
            "#0053A0" if team == home_team else "#DA291C"
        )

        for outcome, marker in outcome_markers.items():
            group = team_data[team_data["outcome"] == outcome]

            if group.empty:
                continue

            ax.scatter(
                group["end_y"],
                group["end_z"],
                s=350,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=1.5,
                zorder=2,
            )

            # Shootouts show attempt order; in-game penalties show minute.
            for _, row in group.iterrows():
                if penalty_type == "Shootout":
                    label = str(row["order"])
                else:
                    label = f'{int(row["minute"])}′'

                ax.text(
                    row["end_y"],
                    row["end_z"],
                    label,
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=3,
                )

    ax.set_xlim(34.5, 45.5)
    ax.set_ylim(-0.25, 4.5)
    ax.axis("off")

    title = (
        "Penalty Shootout"
        if penalty_type == "Shootout"
        else "In-game Penalties"
    )

    fig.suptitle(
        f"{home_team} vs {away_team} — {title}",
        fontsize=14,
        fontweight="bold",
    )

    # Outcome legend.
    legend_items = [
        Line2D(
            [0], [0],
            marker=marker,
            color="black",
            linestyle="None",
            markersize=8,
            label=outcome,
        )
        for outcome, marker in [
            ("Goal", "o"),
            ("Saved", "X"),
            ("Miss / Post", "D"),
        ]
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    return fig


def plot_goalkeeper_shots_faced(events):
    # Keep only shots.
    shots = events[
        events["type"] == "Shot"
    ].copy()

    # Parse shot-end coordinates.
    shots["end_location"] = shots["shot_end_location"].apply(
        parse_location
    )

    shots = shots.dropna(
        subset=["end_location"]
    ).copy()

    # Keep valid goalmouth coordinates.
    shots = shots[
        shots["end_location"].apply(
            lambda loc: isinstance(loc, (list, tuple, np.ndarray))
            and len(loc) >= 3
        )
    ].copy()

    if shots.empty:
        return None

    # Extract goalmouth coordinates.
    shots["end_y"] = shots["end_location"].apply(
        lambda loc: loc[1]
    )

    shots["end_z"] = shots["end_location"].apply(
        lambda loc: loc[2]
    )

    outcomes = shots["shot_outcome"].fillna("Unknown")

    # Create figure.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_goal(ax)

    # Shot outcome colors.
    outcome_colors = {
        "Goal": "#2E8B57",
        "Saved": "#2878B5",
        "Blocked": "#E69F00",
        "Off T": "#888888",
        "Wayward": "#888888",
        "Post": "#9B59B6",
        "Saved to Post": "#2878B5",
        "Saved Off Target": "#2878B5",
        "Deflected": "#888888",
        "Unknown": "#888888"
    }

    # Plot shots individually.
    for outcome, color in outcome_colors.items():
        mask = outcomes == outcome

        if not mask.any():
            continue

        ax.scatter(
            shots.loc[mask, "end_y"],
            shots.loc[mask, "end_z"],
            s=180,
            c=color,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.85,
            label=outcome,
            zorder=3
        )

    ax.set_xlim(34.5, 45.5)
    ax.set_ylim(-0.25, 4.5)
    ax.axis("off")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=9
    )

    fig.tight_layout()

    return fig


player_analysis = player_mapping_df.merge(
    rosters_df,
    left_on=['player_abbr', 'jersey_number', 'team'],
    right_on=['Player', 'Number', 'Team'],
    how='left'
)

player_analysis = player_analysis.rename(columns={
    'player': 'statsbomb_player',
    'player_abbr': 'player_name',
    'Pos': 'position'
})

player_analysis = player_analysis[
    ['statsbomb_player', 'player_name', 'jersey_number', 'team', 'position']
].drop_duplicates()




app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style("""
            .team-stats-container {
                max-width: 1400px;
                width: calc(100% - 48px);
                margin: 40px auto;
            }
            

            .stat-category {
                background-color: #F7F5F6;
                border: 1px solid #E5DDE1;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
            }

            .stat-category-title {
                color: #56042C;
                font-size: 20px;
                font-weight: 700;
                margin: 0 0 20px 0;
                padding-bottom: 12px;
                border-bottom: 1px solid #DED3D9;
            }

            @media (max-width: 600px) {
                .stat-category {
                    padding: 16px;
                }
            }

            /* Player Stats category selector */
            .player-stats-selector .shiny-input-radiogroup {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
            }

            .player-stats-selector .shiny-options-group {
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
                padding: 5px;
                background-color: #F3F4F6;
                border-radius: 10px;
                width: fit-content;
                max-width: 100%;
            }

            .player-stats-selector .shiny-options-group label {
                margin: 0;
                padding: 9px 18px;
                border-radius: 7px;
                cursor: pointer;
                font-weight: 600;
                color: #4B5563;
                transition: background-color 0.15s ease, color 0.15s ease;
            }

            .player-stats-selector .shiny-options-group label:has(input:checked) {
                background-color: #56042C;
                color: white;
            }

            .player-stats-selector .shiny-options-group input {
                position: absolute;
                opacity: 0;
                width: 1px;
                height: 1px;
            }

            .player-stats-selector .shiny-options-group label:focus-within {
                outline: 2px solid #56042C;
                outline-offset: 2px;
            }

            @media (max-width: 500px) {
                .player-stats-selector .shiny-options-group {
                    width: 100%;
                }

                .player-stats-selector .shiny-options-group label {
                    flex: 1;
                    text-align: center;
                    padding: 9px 8px;
                }
                .player-stats-selector .control-label {
                    display: block;
                    margin-bottom: 10px;
                    text-align: center;
                }
            }

            .player-stats-section {
                width: 100%;
                max-width: 1300px;
                box-sizing: border-box;
                margin: 20px auto 32px auto;
            }

            .player-stats-section h2 {
                margin-bottom: 22px;
            }

            .player-team-filter {
                max-width: 320px;
                margin: 18px 0 14px 0;
            }

            .goalkeeper-stats-section {
                border-top: 1px solid #E5E7EB;
                padding-top: 24px;
                margin-top: 36px;
            }

            .goalkeeper-stats-section .form-group {
                margin-bottom: 14px;
            }

            .tournament-stats-table-wrapper {
                width: 100%;
                max-width: 1300px;
                margin: 0 auto;
                overflow-x: auto;
                box-sizing: border-box;
                padding: 8px 0;
            }

            .tournament-stats-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                table-layout: auto;
                white-space: nowrap;
                text-align: center;
                background: white;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }

            .tournament-stats-table th {
                background-color: var(--team-color);
                color: white;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.3px;
                padding: 15px 18px;
                border-bottom: 3px solid var(--team-color);
            }

            .tournament-stats-table td {
                padding: 16px 18px;
                font-size: 15px;
                font-weight: 500;
                color: #333333;
                border-bottom: 1px solid #EEEEEE;
            }

            .tournament-stats-table tbody tr:last-child td {
                border-bottom: none;
            }

            .tournament-stats-table tbody tr:hover {
                background-color: color-mix(in srgb, var(--team-color) 10%, white);
            }

            .tournament-stats-table th:first-child {
                border-top-left-radius: 9px;
            }

            .tournament-stats-table th:last-child {
                border-top-right-radius: 9px;
            }

            .match-log-table-wrapper {
                width: 100%;
                max-width: 1300px;
                margin: 0 auto;
                overflow-x: auto;
                box-sizing: border-box;
                padding: 8px 0;
            }

            .match-log-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                table-layout: auto;
                white-space: nowrap;
                text-align: center;
                background: white;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }

            .match-log-table th {
                background-color: var(--team-color);
                color: white;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.3px;
                padding: 15px 18px;
                border-bottom: 3px solid var(--team-color);
            }

            .match-log-table td {
                padding: 14px 18px;
                font-size: 14px;
                font-weight: 500;
                color: #333333;
                border-bottom: 1px solid #EEEEEE;
            }

            .match-log-table tbody tr:last-child td {
                border-bottom: none;
            }

            .match-log-table tbody tr:nth-child(even) {
                background-color: #F8F9FA;
            }

            .match-log-table tbody tr:hover {
                background-color: color-mix(in srgb, var(--team-color) 10%, white);
            }
            
            .match-player-table-wrapper {
                width: 100%;
                max-width: 1300px;
                margin: 0 auto;
                overflow-x: auto;
                box-sizing: border-box;
                padding: 8px 0;
            }

            .match-player-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                table-layout: auto;
                white-space: nowrap;
                text-align: center;
                background: white;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }

            .match-player-table th {
                background-color: var(--team-color);
                color: white;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.3px;
                padding: 15px 18px;
                border-bottom: 3px solid var(--team-color);
            }

            .match-player-table td {
                padding: 14px 18px;
                font-size: 14px;
                font-weight: 500;
                color: #333333;
                border-bottom: 1px solid #EEEEEE;
            }

            .match-player-table tbody tr:last-child td {
                border-bottom: none;
            }

            .match-player-table tbody tr:nth-child(even) {
                background-color: #F8F9FA;
            }

            .match-player-table tbody tr:hover {
                background-color: color-mix(in srgb, var(--team-color) 10%, white);
            }

            
        """)
    ),
    ui.h1("2022 World Cup Dashboard"),

    ui.navset_tab(

        ui.nav_panel(
            "Match Analysis",

            ui.div(
                ui.div(
                    ui.input_select(
                        "stage",
                        "Tournament Stage",
                        choices=[]
                    ),
                    ui.input_select(
                        "match",
                        "Match",
                        choices=[]
                    ),
                    style="display: flex; gap: 20px; justify-content: center;"
                ),
                style="margin-bottom: 20px;"
            ),

            ui.output_ui("match_header"),
            ui.navset_tab(
                # Team Stats tab
                ui.nav_panel(
                    "Team Stats",

                    ui.div(
                        ui.h2(
                            "Team Stats",
                            style="""
                                color: #56042C;
                                font-size: 26px;
                                font-weight: 700;
                                margin-bottom: 24px;
                                text-align: center;
                            """
                        ),

                        # Attacking panel
                        ui.div(
                            ui.h3("Attacking", class_="stat-category-title"),
                            ui.output_ui("xg_comparison"),
                            ui.output_ui("shots_comparison"),
                            ui.output_ui("shots_on_target_comparison"),
                            ui.output_ui("corners_comparison"),
                            ui.output_ui("final_third_entries_comparison"),
                            ui.output_ui("penalties_won_comparison"),
                            class_="stat-category"
                        ),

                        # Ball Progression panel
                        ui.div(
                            ui.h3("Ball Progression", class_="stat-category-title"),
                            ui.output_ui("pass_completion_comparison"),
                            ui.output_ui("field_tilt_comparison"),
                            class_="stat-category"
                        ),

                        # Defending panel
                        ui.div(
                            ui.h3("Defending", class_="stat-category-title"),
                            ui.output_ui("tackles_comparison"),
                            ui.output_ui("interceptions_comparison"),
                            ui.output_ui("blocks_comparison"),
                            ui.output_ui("clearances_comparison"),
                            ui.output_ui("aerial_wins_comparison"),
                            ui.output_ui("own_goal_comparison"),
                            class_="stat-category"
                        ),

                        # Discipline panel
                        ui.div(
                            ui.h3("Discipline", class_="stat-category-title"),
                            ui.output_ui("fouls_comparison"),
                            ui.output_ui("yellow_cards_comparison"),
                            ui.output_ui("red_cards_comparison"),
                            class_="stat-category"
                        ),

                        class_="team-stats-container"
                    )
                ),

                # Lineups tab — content comes next
                ui.nav_panel(
                    "Lineups",
                    ui.div(
                        ui.h2(
                            "Lineups",
                            style="color: #56042C; text-align: center;"
                        ),
                        ui.output_ui("lineup_pitch"),
                        class_="team-stats-container"
                    )
                ),

                ui.nav_panel(
                    "Player Stats",
                    ui.div(
                        ui.h2(
                            "Outfield Player Stats",
                            style="color: #56042C; text-align: center;"
                        ),
                        ui.div(
                            ui.input_radio_buttons(
                                "player_stats_type",
                                "Stat Category",
                                choices={
                                    "Standard": "Standard",
                                    "Attacking": "Attacking",
                                    "Midfield": "Possession & Distribution",
                                    "Defending": "Defending"
                                },
                                selected="Standard",
                                inline=True
                            ),
                            class_="player-stats-selector"
                        ),
                        ui.div(
                            ui.input_select(
                                "player_team_filter",
                                "Filter by Team",
                                choices={"All Teams": "All Teams"},
                                selected="All Teams"
                            ),
                            class_="player-team-filter"
                        ),
                        ui.output_ui("standard_player_table"),
                        class_="player-stats-section"
                    ),
                    ui.div(
                        ui.h2(
                            "Goalkeeper Stats",
                            style="color: #56042C; text-align: center; margin-top: 30px;"
                        ),
                        ui.div(
                            ui.input_select(
                                "keeper_team_filter",
                                "Filter by Team",
                                choices={"All Teams": "All Teams"},
                                selected="All Teams"
                            ),
                            class_="player-team-filter"
                        ),
                        ui.output_ui("keeper_player_table"),
                        class_="team-stats-container"
                    )
                    
                    
                ),

                ui.nav_panel(
                    "Match Visuals",
                    ui.card(
                        ui.card_header("xT Momentum"),
                        ui.output_ui("xt_momentum_chart"),
                        padding = 2
                    ),
                    ui.card(
                        ui.card_header("Cumulative xG"),
                        ui.output_ui("cumulative_xg_chart"),
                        padding=2
                    ),
                    ui.card(
                        ui.card_header("Shot Map"),
                        ui.output_ui("shot_map_chart"),
                        padding=2
                    ),
                    ui.hr(),
                    ui.h4("Penalty Analysis"),

                    ui.input_select(
                        "penalty_type",
                        "Penalty Type",
                        choices={
                            "In-game": "In-game Penalties",
                            "Shootout": "Penalty Shootouts",
                        },
                        selected="In-game",
                    ),

                    ui.output_ui("penalty_visual"),
                    ui.output_ui("penalty_table"),
                )
            )
        ),

        ui.nav_panel(
            "Player Analysis",


            # Shared player selection
            ui.div(
                ui.div(
                    ui.input_select(
                        "player_country",
                        "Country",
                        choices=[]
                    ),
                    ui.input_select(
                        "player_name",
                        "Player",
                        choices=[]
                    ),
                    style="display: flex; gap: 20px; justify-content: center;"
                ),
                style="margin-bottom: 20px;"
            ),

            ui.output_ui("player_header"),
            ui.output_ui("player_summary"),

            # Player analysis tabs
            ui.navset_tab(
                ui.nav_panel(
                    "Player Overview",

                    # Existing player header and summary
                

                    ui.h2(
                        "Tournament Statistics",
                        style="color: #56042C; text-align: center; margin-top: 30px;"
                    ),
                    ui.output_ui("player_category_selector"),
                    ui.div(
                        ui.div(
                            ui.output_ui("player_tournament_table"),
                            class_="player-tournament-grid"
                        ),
                        class_="player-tournament-table-container"
                    ),

                    ui.h2(
                        "Match-by-Match Performance",
                        style="color: #56042C; text-align: center; margin-top: 40px;"
                    ),

                    ui.div(
                        ui.div(
                            ui.output_ui("player_match_log"),
                            class_="player-match-grid"
                        ),
                        class_="player-match-log-container"
                    ),
                ),

                ui.nav_panel(
                    "Player Visualizations",

                    ui.card(
                        ui.card_header("Player Event Heat Map"),
                        ui.output_ui("player_heat_map"),
                        padding=2
                    ),

                    ui.card(
                        ui.output_ui("player_visualization_header"),
                        ui.output_ui("player_visualization_map"),
                        padding=2
                    ),

                    ui.card(
                        ui.card_header("Player Pass Map"),
                        ui.output_ui("player_pass_map"),
                        padding=2
                    )
                ),

                ui.nav_panel(
                    "Similar Performances",
                    ui.card(
                        ui.card_header("Similar Tournament Performances"),
                        ui.output_ui("similar_performances"),
                        padding=2
                    )
                )    
            )
        ),

        ui.nav_panel(
            "Team Analysis",
            ui.h2(
                "Team Analysis",
                style="color: #56042C; text-align: center;"
            ),
            ui.div(
                ui.div(
                    ui.input_select(
                        "team_country",
                        "Country",
                        choices=[]
                    ),
                    style="display: flex; gap: 20px; justify-content: center;"
                ),
                style="margin-bottom: 20px;"
            ),

            ui.output_ui("team_analysis_header"),

            ui.navset_tab(
                    ui.nav_panel("Roster"),
                    ui.nav_panel("Matches"),
                    ui.nav_panel("Leaderboards"),
                    ui.nav_panel("Team Visualizations")
                )          
        
        )

    )

)

def server(input, output, session):
    ui.update_select(
        "stage",
        choices=schedule["competition_stage"].unique().tolist()
    )

    @reactive.effect
    def update_matches():
        filtered = schedule[
            schedule["competition_stage"] == input.stage()
        ]

        choices = dict(
            zip(
                filtered["match_id"],
                filtered["game_classification"]
            )
        )

        ui.update_select(
            "match",
            choices=choices
        )

    @reactive.effect
    @reactive.event(input.match)
    def update_player_team_filter():
        match = selected_match()

        if match is None:
            return

        teams = [
            match["home_team"],
            match["away_team"]
        ]

        choices = {
            "All Teams": "All Teams",
            **{team: team for team in teams}
        }

        ui.update_select(
            "player_team_filter",
            choices=choices,
            selected="All Teams"
        )
        ui.update_select(
            "keeper_team_filter",
            choices=choices,
            selected="All Teams"
        )

    @reactive.effect
    def update_player_countries():
        countries = sorted(player_analysis['team'].unique())

        ui.update_select(
            "player_country",
            choices=countries,
            selected=countries[0]
        )

    @reactive.effect
    def update_team_countries():
        countries = sorted(team_stats["Team"].unique())

        ui.update_select(
            "team_country",
            choices=countries,
            selected=countries[0]
        )

    @reactive.effect
    @reactive.event(input.player_country)
    def update_player_choices():
        country_players = player_analysis[
            player_analysis['team'] == input.player_country()
        ]

        players = sorted(country_players['player_name'].unique())

        ui.update_select(
            "player_name",
            choices=players,
            selected=players[0]
        )

    @reactive.calc
    def selected_match():
        if not input.match():
            return None

        return schedule[
            schedule["match_id"] == int(input.match())
        ].iloc[0]

    @reactive.calc
    def selected_team_stats():
        if not input.match():
            return None

        return team_stats[
            team_stats["match_id"] == int(input.match())
        ]

    @reactive.calc
    def selected_starters():
        if not input.match():
            return None

        return starters_df[
            (starters_df["match_id"] == int(input.match()))
            & (starters_df["Starter"] == True)
        ]

    @reactive.calc
    def selected_sub_entries():
        match = selected_match()

        if match is None:
            return None

        match_id = int(input.match())

        subs = sub_entries[
            sub_entries["match_id"] == match_id
        ].copy()

        mapping = (
            player_mapping_df[
                ["player", "player_abbr", "team"]
            ]
            .drop_duplicates()
        )

        # Map the player coming OFF
        off_mapping = mapping.rename(
            columns={
                "player": "statsbomb_off",
                "player_abbr": "player_off_name",
            }
        )

        subs = subs.merge(
            off_mapping,
            left_on=["player", "team"],
            right_on=["statsbomb_off", "team"],
            how="left",
        )

        # Map the player coming ON
        on_mapping = mapping.rename(
            columns={
                "player": "statsbomb_on",
                "player_abbr": "player_on_name",
            }
        )

        subs = subs.merge(
            on_mapping,
            left_on=["substitution_replacement", "team"],
            right_on=["statsbomb_on", "team"],
            how="left",
        )

        return subs

    @reactive.calc
    def selected_lineup_data():
        match = selected_match()

        if match is None:
            return None

        starters = selected_starters()

        if starters is None:
            return None

        return {
            "starters": starters,
            "substitutes": starters_df[
                (starters_df["match_id"] == int(input.match()))
                & (starters_df["Starter"] == False)
            ],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_formation": str(match["home_formation"]),
            "away_formation": str(match["away_formation"]),
        }

    @reactive.calc
    def selected_player():
        player = input.player_name()
        country = input.player_country()

        if not player or not country:
            return None

        result = player_analysis[
            (player_analysis['player_name'] == player) &
            (player_analysis['team'] == country)
        ]

        if result.empty:
            return None

        return result.iloc[0]

    @reactive.calc
    def player_tournament_summary():
        player = selected_player()

        if player is None:
            return None

        statsbomb_name = player['statsbomb_player']

        if player['position'] == 'GK':
            player_stats = keeper_stats[
                keeper_stats['Player'] == statsbomb_name
            ]
        else:
            player_stats = standard_stats[
                standard_stats['Player'] == statsbomb_name
            ]

        minutes = player_stats['Minutes'].sum()
        appearances = player_stats['game_id'].nunique()

        starts = starters_df[
            (starters_df['player'] == statsbomb_name) &
            (starters_df['Starter'] == True)
        ]['match_id'].nunique()

        return {
            'minutes': minutes,
            'appearances': appearances,
            'starts': starts
        }

    @reactive.calc
    def selected_player_stats():
        player = selected_player()

        if player is None:
            return pd.DataFrame()

        statsbomb_name = player['statsbomb_player']

        if player['position'] == 'GK':
            stats = keeper_stats
        else:
            category = input.player_tournament_type()

            if category == "Standard":
                stats = standard_stats
            elif category == "Attacking":
                stats = attacker_stats
            elif category == "Midfield":
                stats = midfielder_stats
            else:
                stats = defender_stats

        player_stats = stats[
            stats['Player'] == statsbomb_name
        ]

        # Remove columns that identify the player or individual match.
        excluded_columns = [
            'Unnamed: 0', 'Player', 'Position', 'Team', 'game_id'
        ]

        stat_columns = [
            col for col in player_stats.columns
            if col not in excluded_columns
        ]

        # Sum cumulative counting statistics.
        totals = player_stats[stat_columns].sum().to_frame().T


        # Calculate tournament-wide pass completion percentage.
        if 'Pass Completion %' in totals.columns:
            total_passes = player_stats['Passes'].sum()

            totals['Pass Completion %'] = (
                (
                    player_stats['Pass Completion %'] *
                    player_stats['Passes']
                ).sum() / total_passes
                if total_passes > 0 else 0
            )

        for col in ['Pass Completion %', 'xG']:
            if col in totals.columns:
                totals[col] = totals[col].round(2)

        return totals.reset_index(drop=True)

    @reactive.calc
    def player_match_log_data():
        player = selected_player()

        if player is None:
            return pd.DataFrame()

        statsbomb_name = player['statsbomb_player']

        # Select dataset and statistics based on player position/category.
        if player['position'] == 'GK':
            stats = keeper_stats
            stat_columns = [
                'Minutes',
                'Shots Faced',
                'Saves',
                'Goals Conceded',
                'Clearances'
            ]
        else:
            category = input.player_tournament_type()

            if category == "Standard":
                stats = standard_stats
                stat_columns = [
                    'Minutes', 'Goals', 'Assists', 'Shots',
                    'Shots on Target', 'Passes', 'Pass Completion %',
                    'Fouls', 'Yellow Cards', 'Red Cards'
                ]
            elif category == "Attacking":
                stats = attacker_stats
                stat_columns = [
                    'Minutes', 'Goals', 'xG', 'Shots',
                    'Shots on Target', 'Assists',
                    'Penalties Won', 'Final Third Entries'
                ]
            elif category == "Midfield":
                stats = midfielder_stats
                stat_columns = [
                    'Minutes', 'Passes', 'Key Passes',
                    'Successful Dribbles', 'Crosses', 'Free Kicks',
                    'Dispossessed', 'Miscontrols', 'Aerial Wins'
                ]
            else:
                stats = defender_stats
                stat_columns = [
                    'Minutes', 'Own Goals', 'Tackles',
                    'Interceptions', 'Blocks', 'Clearances',
                    'Fouls', 'Yellow Cards', 'Red Cards',
                    'Penalty Conceded'
                ]

        # Filter selected player.
        player_games = stats[
            stats['Player'] == statsbomb_name
        ].copy()

        # Join schedule information.
        game_log = player_games.merge(
            schedule,
            left_on='game_id',
            right_on='match_id',
            how='left'
        )

        # Determine opponent and result from player's perspective.
        def match_details(row):
            if row['Team'] == row['home_team']:
                opponent = row['away_team']
                team_score = row['home_score']
                opponent_score = row['away_score']
                venue = 'Home'
            else:
                opponent = row['home_team']
                team_score = row['away_score']
                opponent_score = row['home_score']
                venue = 'Away'

            if team_score > opponent_score:
                result = 'W'
            elif team_score < opponent_score:
                result = 'L'
            else:
                result = 'D'

            return pd.Series({
                'Opponent': opponent,
                'Venue': venue,
                'Result': f"{result} {team_score}–{opponent_score}"
            })

        details = game_log.apply(match_details, axis=1)
        game_log = pd.concat([game_log, details], axis=1)

        # Convert date to datetime.
        game_log['match_date'] = pd.to_datetime(game_log['match_date'])

        # Sort chronologically before formatting.
        game_log = game_log.sort_values('match_date')

        # Format date for display.
        game_log['Date'] = game_log['match_date'].dt.strftime('%b %d, %Y')

        # Select displayed columns.
        game_log = game_log[
            ['Date', 'Opponent', 'Venue', 'Result'] + stat_columns
        ]

        return game_log.reset_index(drop=True)
    
    def create_stat_comparison(metric, label, decimals=0):
        stats = selected_team_stats()

        if stats is None:
            return ui.div()

        home_team = stats.iloc[0]["Team"]
        away_team = stats.iloc[1]["Team"]

        home_value = stats.iloc[0][metric]
        away_value = stats.iloc[1][metric]

        home_color = team_colors[home_team]
        away_color = team_colors[away_team]

        total = home_value + away_value

        if total == 0:
            home_share = 0.5
            away_share = 0.5
        else:
            home_share = home_value / total
            away_share = away_value / total

        number_format = f"{{:.{decimals}f}}"

        return ui.div(
            ui.div(
                ui.div(
                    ui.p(
                        home_team,
                        style=f"""
                            font-size: 20px;
                            font-weight: 600;
                            margin-bottom: 5px;
                            color: {home_color};
                            text-shadow: 0 0 1px black;
                        """
                    ),
                    ui.p(
                        number_format.format(home_value),
                        style=f"""
                            font-size: 32px;
                            font-weight: 700;
                            margin: 0;
                            color: {home_color};
                            text-shadow: 0 0 1px black;
                        """
                    ),
                    style="text-align: center;"
                ),
                ui.div(
                    ui.div(
                        style=f"""
                            width: {home_share * 100}%;
                            height: 12px;
                            background-color: {home_color};
                            border-right: 2px solid white;
                        """
                    ),
                    ui.div(
                        style=f"""
                            width: {away_share * 100}%;
                            height: 12px;
                            background-color: {away_color};
                        """
                    ),
                    style="""
                        display: flex;
                        width: 100%;
                        border-radius: 999px;
                        overflow: hidden;
                        border: 1px solid #D0D0D0;
                    """
                ),
                ui.div(
                    ui.p(
                        away_team,
                        style=f"""
                            font-size: 20px;
                            font-weight: 600;
                            margin-bottom: 5px;
                            color: {away_color};
                            text-shadow: 0 0 1px black;
                        """
                    ),
                    ui.p(
                        number_format.format(away_value),
                        style=f"""
                            font-size: 32px;
                            font-weight: 700;
                            margin: 0;
                            color: {away_color};
                            text-shadow: 0 0 1px black;
                        """
                    ),
                    style="text-align: center;"
                ),
                style="""
                    display: grid;
                    grid-template-columns: 1fr 2fr 1fr;
                    align-items: center;
                    gap: 25px;
                """
            ),
            ui.p(
                label,
                style="""
                    text-align: center;
                    font-size: 15px;
                    font-weight: 600;
                    margin-top: 12px;
                    color: #666;
                """
            ),
            style="""
                max-width: 950px;
                width: 100%;
                box-sizing: border-box;
                margin: 25px auto;
            """
        )

    @render.ui
    def xg_comparison():
        return create_stat_comparison(
            "xG",
            "Expected Goals (xG)",
            decimals=2
        )
    
    @render.ui
    def shots_comparison():
        return create_stat_comparison(
            "Shots",
            "Shots",
            decimals=0
        )

    @render.ui
    def shots_on_target_comparison():
        return create_stat_comparison(
            "Shots On Target",
            "Shots on Target",
            decimals=0
        )

    @render.ui
    def corners_comparison():
        return create_stat_comparison(
            "Corners",
            "Corners",
            decimals=0
        )

    @render.ui
    def final_third_entries_comparison():
        return create_stat_comparison(
            "Final Third Entries",
            "Final Third Entries",
            decimals=0
        )


    @render.ui
    def penalties_won_comparison():
        return create_stat_comparison(
            "Penalties Won",
            "Penalties Won",
            decimals=0
        )


    @render.ui
    def pass_completion_comparison():
        return create_stat_comparison(
            "Pass Completion %",
            "Pass Completion %",
            decimals=1
        )


    @render.ui
    def field_tilt_comparison():
        return create_stat_comparison(
            "Field Tilt",
            "Field Tilt",
            decimals=1
        )


    @render.ui
    def tackles_comparison():
        return create_stat_comparison(
            "Tackles",
            "Tackles",
            decimals=0
        )


    @render.ui
    def interceptions_comparison():
        return create_stat_comparison(
            "Interceptions",
            "Interceptions",
            decimals=0
        )


    @render.ui
    def blocks_comparison():
        return create_stat_comparison(
            "Blocks",
            "Blocks",
            decimals=0
        )


    @render.ui
    def clearances_comparison():
        return create_stat_comparison(
            "Clearances",
            "Clearances",
            decimals=0
        )


    @render.ui
    def aerial_wins_comparison():
        return create_stat_comparison(
            "Aerial Wins",
            "Aerial Wins",
            decimals=0
        )


    @render.ui
    def fouls_comparison():
        return create_stat_comparison(
            "Fouls",
            "Fouls",
            decimals=0
        )


    @render.ui
    def yellow_cards_comparison():
        return create_stat_comparison(
            "Yellow Card",
            "Yellow Cards",
            decimals=0
        )


    @render.ui
    def red_cards_comparison():
        return create_stat_comparison(
            "Red Card",
            "Red Cards",
            decimals=0
        )

    @render.ui
    def own_goal_comparison():
        return create_stat_comparison(
            "Own Goals",
            "Own Goals",
            decimals=0
        )

    @render.ui
    def player_visualization_header():
        player = selected_player()

        if player is None:
            return ui.card_header("Player Shot Map")

        if player["position"] == "GK":
            return ui.card_header("Goalkeeper Shots Faced")

        return ui.card_header("Player Shot Map")

    @render.ui
    def team_analysis_header():
        country = input.team_country()

        if not country:
            return ui.div()

        team_color = team_cols.loc[
            team_cols['Country'] == country,
            'primary_color_hex'
        ].iloc[0]

        finish = team_finishes.get(country, "Unknown")

        return ui.div(
            ui.p(
                country,
                style="""
                    font-size: 45px;
                    font-weight: 700;
                    margin-bottom: 5px;
                """
            ),

            ui.p(
                finish,
                style=f"""
                    font-size: 20px;
                    font-weight: 700;
                    color: {team_color};
                    margin-top: 0;
                """
            ),

            style=f"""
                    max-width: 1300px;
                    width: 100%;
                    box-sizing: border-box;
                    margin: 30px auto;
                    padding: 40px 50px;
                    text-align: center;
                    border: 4px solid {team_color};
                    border-radius: 16px;
                """
        )


    @render.ui
    def player_visualization_map():
        player = selected_player()

        if player is None:
            return ui.div()

        if player["position"] == "GK":
            return ui.output_ui("goalkeeper_shot_map")

        return ui.output_ui("player_shot_map")

 
    @render.ui
    def lineup_pitch():
        lineup = selected_lineup_data()

        if lineup is None:
            return ui.p("Select a match to view the lineups.")

        starters = lineup["starters"]
        home_team = lineup["home_team"]
        away_team = lineup["away_team"]
        home_formation = lineup["home_formation"]
        away_formation = lineup["away_formation"]

        formations = {
            home_team: lineup["home_formation"],
            away_team: lineup["away_formation"]
        }

        # Draw the soccer pitch
        pitch = Pitch(
            pitch_color="grass",
            line_color="White",
            stripe=True
        )

        fig, ax = pitch.draw(figsize=(12, 7))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Add each player's vertical ordering
        players = starters.copy()
        players["pitch_order"] = players["position_id"].map(
            pitch_order
        )

        # Draw both teams
        for team in [home_team, away_team]:
            formation = formations[team]
            formation_lines = list(formation)

            team_players = (
                players[players["team"] == team]
                .sort_values("position_id")
                .reset_index(drop=True)
            )

            # Separate goalkeeper from outfield players
            goalkeeper = team_players[
                team_players["position_id"] == 1
            ]

            outfield = team_players[
                team_players["position_id"] != 1
            ].copy()

            if goalkeeper.empty:
                continue

            is_home = team == home_team
            color = team_colors.get(team, "#56042C")

            # Goalkeeper
            gk_x = 8 if is_home else 112
            gk_jersey = goalkeeper.iloc[0]["jersey_number"]

            ax.scatter(
                gk_x, 40,
                s=500,
                color=color,
                edgecolors="white",
                zorder=3
            )

            ax.text(
                gk_x, 40,
                str(gk_jersey),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                zorder=4
            )

            # Select horizontal coordinates for this formation
            x_coordinates = xCords.get(
                str(len(formation_lines))
            )

            if x_coordinates is None:
                continue

            # Plot each outfield line
            for index, line_size in enumerate(formation_lines):
                width = int(line_size)


                y_coordinates = yCords.get(str(width))

                if y_coordinates is None:
                    continue

                # Take the players for this line and order them
                line_players = (
                    outfield.iloc[:width]
                    .sort_values("pitch_order")
                    .reset_index(drop=True)
                )

                outfield = outfield.iloc[width:].reset_index(
                    drop=True
                )

                for j, (_, player) in enumerate(
                    line_players.iterrows()
                ):
                    if j >= len(y_coordinates):
                        break

                    x = x_coordinates[index]
                    y = y_coordinates[j]

                    # Mirror the away team across the pitch
                    if not is_home:
                        x = 120 - x
                        y = 80 - y

                    jersey = player["jersey_number"]

                    ax.scatter(
                        x, y,
                        s=500,
                        color=color,
                        edgecolors="white",
                        zorder=3
                    )

                    ax.text(
                        x, y,
                        str(jersey),
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        zorder=4
                    )

        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        def create_starting_list(team):
            team_starters = (
                starters[starters["team"] == team]
                .sort_values("position_id")
            )

            rows = []

            for _, player in team_starters.iterrows():
                rows.append(
                    ui.div(
                        ui.span(
                            str(player["jersey_number"]),
                            style=(
                                "display: inline-flex;"
                                "align-items: center;"
                                "justify-content: center;"
                                "min-width: 30px;"
                                "height: 30px;"
                                "border-radius: 50%;"
                                f"background-color: {team_colors.get(team, '#56042C')};"
                                "color: white;"
                                "font-weight: 700;"
                                "margin-right: 10px;"
                            )
                        ),
                        ui.span(
                            str(player["player_abbr"]),
                            style="font-weight: 600;"
                        ),
                        ui.span(
                            str(player["position"]),
                            style=(
                                "margin-left: auto;"
                                "font-size: 12px;"
                                "color: #777;"
                                "text-align: right;"
                            )
                        ),
                        style=(
                            "display: flex;"
                            "align-items: center;"
                            "gap: 6px;"
                            "padding: 8px 0;"
                            "border-bottom: 1px solid #E5DDE1;"
                        )
                    )
                )

            return ui.div(*rows)

        
        def create_substitution_list(team):
            subs = selected_sub_entries()

            if subs is None:
                return ui.div()

            team_subs = subs[
                subs["team"] == team
            ].copy()

            # Show a message when no substitutions occurred
            if team_subs.empty:
                return ui.div(
                    ui.h4("Substitutions"),
                    ui.p(
                        "No substitutions recorded.",
                        style="color: #6B7280;"
                    )
                )

            # Use the full roster to look up jersey numbers
            roster = rosters_df[
                rosters_df["Team"] == team
            ]

            jersey_numbers = dict(
                zip(roster["Player"], roster["Number"])
            )

            items = []

            for _, sub in team_subs.sort_values("minute").iterrows():
                off_name = sub["player_off_name"]
                on_name = sub["player_on_name"]

                # Fall back to the original event name if unmapped
                if pd.isna(off_name):
                    off_name = sub["player"]

                if pd.isna(on_name):
                    on_name = sub["substitution_replacement"]

                off_number = jersey_numbers.get(off_name, "—")
                on_number = jersey_numbers.get(on_name, "—")
                minute = int(sub["minute"])

                items.append(
                    ui.div(
                        # Player coming off
                        ui.div(
                            ui.div(
                                str(off_number),
                                style=(
                                    f"background-color: {team_colors.get(team, '#56042C')}; "
                                    "color: white; width: 32px; height: 32px; "
                                    "border-radius: 50%; display: flex; "
                                    "align-items: center; justify-content: center; "
                                    "font-weight: bold; flex-shrink: 0;"
                                )
                            ),
                            ui.div(
                                ui.div(
                                    off_name,
                                    style="font-weight: 600;"
                                ),
                                ui.div(
                                    "OFF",
                                    style="font-size: 11px; color: #6B7280;"
                                )
                            ),
                            style=(
                                "display: flex; align-items: center; "
                                "gap: 8px; flex: 1; min-width: 0;"
                            )
                        ),

                        # Substitution minute
                        ui.div(
                            f"{minute}′",
                            style=(
                                "font-weight: 700; color: #56042C; "
                                "white-space: nowrap;"
                            )
                        ),

                        # Player coming on
                        ui.div(
                            ui.div(
                                str(on_number),
                                style=(
                                    f"background-color: {team_colors.get(team, '#56042C')}; "
                                    "color: white; width: 32px; height: 32px; "
                                    "border-radius: 50%; display: flex; "
                                    "align-items: center; justify-content: center; "
                                    "font-weight: bold; flex-shrink: 0;"
                                )
                            ),
                            ui.div(
                                ui.div(
                                    on_name,
                                    style="font-weight: 600;"
                                ),
                                ui.div(
                                    "ON",
                                    style="font-size: 11px; color: #6B7280;"
                                )
                            ),
                            style=(
                                "display: flex; align-items: center; "
                                "gap: 8px; flex: 1; min-width: 0;"
                            )
                        ),

                        style=(
                            "display: flex; align-items: center; "
                            "gap: 12px; margin-bottom: 12px; "
                            "padding: 10px; border: 1px solid #E5E7EB; "
                            "border-radius: 8px;"
                        )
                    )
                )

            return ui.div(
                ui.h4("Substitutions"),
                *items
            )



        image = ui.HTML(
            f'<img src="{image_uri}" '
            'style="width:100%; max-width:1000px; '
            'display:block; margin:auto;">'
        )
        # Team header styling
        home_color = team_colors.get(home_team, "#56042C")
        away_color = team_colors.get(away_team, "#56042C")

        home_formation_display = "-".join(
            list(home_formation)
        )

        away_formation_display = "-".join(
            list(away_formation)
        )

        team_header = ui.div(
            # Home team
            ui.div(
                ui.div(
                    home_team,
                    style=(
                        f"color: {home_color}; "
                        "font-size: 30px; "
                        "font-weight: 700;"
                    )
                ),
                ui.div(
                    home_formation_display,
                    style=(
                        "color: #6B7280; "
                        "font-size: 17px; "
                        "margin-top: 2px;"
                    )
                ),
                style="flex: 1; text-align: left;"
            ),

            # Center label
            ui.div(
                "VS",
                style=(
                    "color: #9CA3AF; "
                    "font-size: 14px; "
                    "font-weight: 700; "
                    "letter-spacing: 2px;"
                )
            ),

            # Away team
            ui.div(
                ui.div(
                    away_team,
                    style=(
                        f"color: {away_color}; "
                        "font-size: 30px; "
                        "font-weight: 700;"
                    )
                ),
                ui.div(
                    away_formation_display,
                    style=(
                        "color: #6B7280; "
                        "font-size: 17px; "
                        "margin-top: 2px;"
                    )
                ),
                style="flex: 1; text-align: right;"
            ),

            style=(
                "display: flex; "
                "align-items: center; "
                "justify-content: space-between; "
                "gap: 16px; "
                "width: 100%; "
                "max-width: 1000px; "
                "margin: 0 auto 18px auto; "
                "padding: 0 8px; "
                "box-sizing: border-box;"
            )
        )

        return ui.div(
            team_header,
            image,

            # Starting XI lists
            ui.div(
                ui.div(
                    ui.h4("Starting XI"),
                    create_starting_list(home_team)
                ),
                ui.div(
                    ui.h4("Starting XI"),
                    create_starting_list(away_team)
                ),
                style=(
                    "display: grid;"
                    "grid-template-columns: 1fr 1fr;"
                    "gap: 32px;"
                    "margin-top: 24px;"
                )
            ),

            # Bench / Non-starters
            ui.div(
                ui.div(
                    create_substitution_list(home_team)
                ),
                ui.div(
                    create_substitution_list(away_team)
                ),
                style=(
                    "display: grid;"
                    "grid-template-columns: 1fr 1fr;"
                    "gap: 32px;"
                    "margin-top: 32px;"
                )
            )

        )

    @render.ui
    def player_team_choices():
        pass

    @output
    @render.ui
    def xt_momentum_chart():
        match = selected_match()

        if match is None:
            return ui.p("Select a match to view its xT momentum.")

        match_id = match["match_id"]

        events = visual_data[
            visual_data["match_id"] == match_id
        ]

        teams = events["team"].dropna().unique()

        if len(teams) < 2:
            return ui.p("Not enough team data to plot this match.")

        home_team, away_team = teams[:2]

        fig = plot_xt_momentum(events, home_team, away_team)
        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; overflow-x:auto; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:1000px; height:auto; display:inline-block;">'
            f'</div>'
        )

    @output
    @render.ui
    def cumulative_xg_chart():
        match = selected_match()

        if match is None:
            return ui.p("Select a match to view cumulative xG.")

        match_id = match["match_id"]
        events = visual_data[
            visual_data["match_id"] == match_id
        ].copy()

        teams = events["team"].dropna().unique()

        if len(teams) < 2:
            return ui.p("Not enough team data to plot this match.")

        home_team, away_team = teams[:2]

        fig = plot_cumulative_xg(events, home_team, away_team)
        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:1000px; height:auto;">'
            f'</div>'
        )

    @output
    @render.ui
    def shot_map_chart():
        match = selected_match()

        if match is None:
            return ui.p("Select a match to view the shot map.")

        match_id = match["match_id"]
        events = visual_data[
            visual_data["match_id"] == match_id
        ].copy()

        teams = events["team"].dropna().unique()

        if len(teams) < 2:
            return ui.p("Not enough team data to plot this match.")

        home_team, away_team = teams[:2]

        fig = plot_shot_map(events, home_team, away_team)
        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:900px; height:auto;">'
            f'</div>'
        )



    
    @render.ui
    def standard_player_table():
        match = selected_match()

        stats_datasets = {
            "Standard": (
                standard_stats,
                [
                    "Player", "Position", "Minutes", "Team",
                    "Goals", "Shots", "Shots on Target",
                    "Passes", "Pass Completion %", "Assists",
                    "Fouls", "Yellow Cards", "Red Cards"
                ]
            ),
            "Attacking": (
                attacker_stats,
                [
                    "Player", "Position", "Minutes", "Team",
                    "Goals", "xG", "Shots", "Shots on Target",
                    "Assists", "Penalties Won",
                    "Final Third Entries"
                ]
            ),
            "Midfield": (
                midfielder_stats,
                [
                    "Player", "Position", "Minutes", "Team",
                    "Passes", "Key Passes", "Crosses",
                    "Successful Dribbles", "Free Kicks",
                    "Dispossessed", "Miscontrols",
                    "Aerial Wins"
                ]
            ),
            "Defending": (
                defender_stats,
                [
                    "Player", "Position", "Minutes", "Team",
                    "Tackles", "Interceptions", "Blocks",
                    "Clearances", "Fouls", "Own Goals",
                    "Yellow Cards", "Red Cards",
                    "Penalty Conceded"
                ]
            )
        }

        stats_type = input.player_stats_type()
        data, columns = stats_datasets[stats_type]

        if match is None:
            return ui.div()

        match_id = int(input.match())

        df = data[data["game_id"] == match_id].copy()

        if "xG" in df.columns:
            df["xG"] = df["xG"].round(2)

        team_filter = input.player_team_filter()

        if team_filter != "All Teams":
            df = df[df["Team"] == team_filter]

        df = df[columns].sort_values(
            ["Team", "Position", "Player"]
        )

        if df.empty:
            return ui.p("No player statistics available.")

        team_color = "#56042C"
        

        return ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        *[
                            ui.tags.th(col)
                            for col in df.columns
                        ]
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            *[
                                ui.tags.td(
                                    f"{value:,.2f}" if isinstance(value, (float, np.floating))
                                    else f"{value:,.0f}" if isinstance(value, (int, np.integer))
                                    else str(value)
                                )
                                for value in row
                            ]
                        )
                        for row in df.itertuples(index=False, name=None)
                    ]
                ),
                class_="match-player-table",
                style=f"--team-color: {team_color};"
            ),
            class_="match-player-table-wrapper"
        )

   
    @output
    @render.ui
    def penalty_visual():
        match = selected_match()
        match_id = match["match_id"]

        match_penalties = penalty_data[
            penalty_data["match_id"] == match_id
        ].copy()

        selected_type = input.penalty_type()

        match_penalties = match_penalties[
            match_penalties["penalty_type"] == selected_type
        ].copy()

        if match_penalties.empty:
            return ui.p("No penalties of this type in the selected match.")

        teams = schedule.loc[
            schedule["match_id"] == match_id,
            ["home_team", "away_team"]
        ]

        if teams.empty:
            return ui.p("Could not identify the teams for this match.")

        home_team = teams.iloc[0]["home_team"]
        away_team = teams.iloc[0]["away_team"]

        fig = plot_penalty_visual(
            match_penalties,
            home_team,
            away_team,
            selected_type,
        )

        if fig is None:
            return ui.p("No usable penalty locations are available.")

        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:900px; height:auto;">'
            f'</div>'
        )

    @output
    @render.ui
    def penalty_table():
        match = selected_match()
        match_id = match["match_id"]

        selected_type = input.penalty_type()

        match_penalties = penalty_data[
            (penalty_data["match_id"] == match_id)
            & (penalty_data["penalty_type"] == selected_type)
        ].copy()

        if match_penalties.empty:
            return ui.p("No penalties to display.")

        # Preserve the actual event order.
        match_penalties = match_penalties.sort_values("index").copy()

        if selected_type == "Shootout":
            first_column = "Attempt"
            match_penalties["Attempt"] = (
                match_penalties.groupby("team").cumcount() + 1
            )
        else:
            first_column = "Minute"
            match_penalties["Minute"] = (
                match_penalties["minute"].astype(int).astype(str) + "'"
            )

        table = match_penalties[
            [first_column, "team", "player", "goalkeeper", "outcome"]
        ].rename(
            columns={
                first_column: "Minute" if selected_type == "In-game" else "Attempt",
                "team": "Team",
                "player": "Penalty Taker",
                "goalkeeper": "Goalkeeper",
                "outcome": "Outcome",
            }
        )

        player_name_map = player_mapping_df.set_index(
        ["team", "player"]
        )["player_abbr"]

        table["Penalty Taker"] = [
            player_name_map.get((team, player), player)
            for team, player in zip(
                match_penalties["team"],
                match_penalties["player"]
            )
        ]

        table["Goalkeeper"] = [
            player_name_map.get((team, player), player)
            for team, player in zip(
                match_penalties["goalkeeper_team"],
                match_penalties["goalkeeper"]
            )
        ]

        # Render a clean HTML table.
        return ui.HTML(
            f"""
            <div style="width:100%; overflow-x:auto; margin-top:12px;">
                {table.to_html(
                    index=False,
                    border=0,
                    classes="penalty-table",
                    na_rep="—",
                    escape=True
                )}
            </div>
            <style>
                .penalty-table {{
                width:100%;
                border-collapse:collapse;
                table-layout:auto;
                font-size:14px;
                text-align:left;
            }}

                .penalty-table thead {{
                    display:table-header-group;
                }}

                .penalty-table tbody {{
                    display:table-row-group;
                }}

                .penalty-table tr {{
                    display:table-row;
                }}

                .penalty-table th,
                .penalty-table td {{
                    display:table-cell;
                    box-sizing:border-box;
                    padding:9px 14px;
                    text-align:left;
                    vertical-align:middle;
                    white-space:nowrap;
                    position:static;
                    transform:none;
                }}

                .penalty-table th {{
                    background-color:#f0f0f0;
                    border-bottom:2px solid #cccccc;
                    font-weight:600;
                }}

                .penalty-table td {{
                    border-bottom:1px solid #e0e0e0;
                }}

                .penalty-table tbody tr:nth-child(even) {{
                    background-color:#f8f8f8;
                }}
            </style>
            """
        )

    @render.ui
    def keeper_player_table():
        columns = [
            "Player", "Position", "Minutes", "Team",
            "Shots Faced", "Saves", "Goals Conceded",
            "Passes", "Clearances"
        ]

        match = selected_match()

        if match is None:
            return ui.div()

        match_id = int(input.match())

        df = keeper_stats[
            keeper_stats["game_id"] == match_id
        ].copy()

        team_filter = input.keeper_team_filter()

        if team_filter != "All Teams":
            df = df[df["Team"] == team_filter]

        df = df[columns].sort_values(
            ["Team", "Player"]
        )

        if df.empty:
            return ui.p("No goalkeeper statistics available.")

        return ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        *[
                            ui.tags.th(col)
                            for col in df.columns
                        ]
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            *[
                                ui.tags.td(
                                    f"{value:,.2f}" if isinstance(value, (float, np.floating))
                                    else f"{value:,.0f}" if isinstance(value, (int, np.integer))
                                    else str(value)
                                )
                                for value in row
                            ]
                        )
                        for row in df.itertuples(index=False, name=None)
                    ]
                ),
                class_="match-player-table",
                style="--team-color: #56042C;"
            ),
            class_="match-player-table-wrapper"
        )


    @reactive.calc
    def goalkeeper_shots_faced():
        player = selected_player()

        if player is None or player["position"] != "GK":
            return pd.DataFrame()

        statsbomb_name = player["statsbomb_player"]
        goalkeeper_team = player["team"]

        # Find matches in which the goalkeeper appeared.
        goalkeeper_matches = keeper_stats[
            keeper_stats["Player"] == statsbomb_name
        ]["game_id"].unique()

        # Get the opponents in those matches.
        match_teams = schedule[
            schedule["match_id"].isin(goalkeeper_matches)
        ][["match_id", "home_team", "away_team"]].copy()

        match_teams["opponent"] = np.where(
            match_teams["home_team"] == goalkeeper_team,
            match_teams["away_team"],
            match_teams["home_team"]
        )

        # Identify shots taken by those opponents.
        opponent_shots = visual_data.merge(
            match_teams[["match_id", "opponent"]],
            on="match_id",
            how="inner"
        )

        opponent_shots = opponent_shots[
            (opponent_shots["team"] == opponent_shots["opponent"]) &
            (opponent_shots["type"] == "Shot")
        ].copy()

        return opponent_shots

    @render.ui
    def goalkeeper_shot_map():
        shots = goalkeeper_shots_faced()

        if shots.empty:
            return ui.p("No shots faced data available for this goalkeeper.")

        fig = plot_goalkeeper_shots_faced(shots)

        if fig is None:
            return ui.p("No valid shot location data available.")

        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:900px; height:auto;">'
            f'</div>'
        )

    @render.ui
    def player_tournament_table():
        stats = selected_player_stats()
        player = selected_player()


        if stats.empty:
            return ui.div()

        team_color = team_colors.get(
            player['team'],
            "#56042C"
        )

        return ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        *[
                            ui.tags.th(col)
                            for col in stats.columns
                        ]
                    )
                ),
                ui.tags.tbody(
                    ui.tags.tr(
                        *[
                            ui.tags.td(
                                f"{value:,.2f}" if isinstance(value, (float, np.floating))
                                else f"{value:,.0f}" if isinstance(value, (int, np.integer))
                                else str(value)
                            )
                            for value in stats.iloc[0]
                        ]
                    )
                ),
                class_="tournament-stats-table",
                style=f"--team-color: {team_color};"
            ),
            class_="tournament-stats-table-wrapper"
        )

    @render.ui
    def player_match_log():
        game_log = player_match_log_data()
        player = selected_player()

        if game_log.empty or player is None:
            return ui.div()

        team_color = team_colors.get(
            player['team'],
            "#56042C"
        )

        # Build table headers.
        headers = ui.tags.tr(
            *[
                ui.tags.th(col)
                for col in game_log.columns
            ]
        )

        # Build table rows.
        rows = []

        for _, row in game_log.iterrows():
            cells = []

            for col, value in row.items():
                if pd.isna(value):
                    display_value = ""
                elif isinstance(value, (float, np.floating)):
                    display_value = f"{value:,.2f}"
                elif isinstance(value, (int, np.integer)):
                    display_value = f"{value:,.0f}"
                else:
                    display_value = str(value)

                cells.append(ui.tags.td(display_value))

            rows.append(ui.tags.tr(*cells))

        return ui.div(
            ui.tags.table(
                ui.tags.thead(headers),
                ui.tags.tbody(*rows),
                class_="match-log-table",
                style=f"--team-color: {team_color};"
            ),
            class_="match-log-table-wrapper"
        )

    @render.ui
    def player_heat_map():
        player = selected_player()

        if player is None:
            return ui.p("Select a player to view their heat map.")

        statsbomb_name = player['statsbomb_player']

        player_events = visual_data[
            visual_data['player'] == statsbomb_name
        ].copy()

        # Create heat map figure.
        fig = plot_player_heat_map(player_events)
        if fig is None:
            return ui.p("No event locations available for this player.")

        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:900px; height:auto;">'
            f'</div>'
        )

    @render.ui
    def player_shot_map():
        player = selected_player()

        if player is None:
            return ui.p("Select a player to view their shot map.")

        statsbomb_name = player["statsbomb_player"]

        player_events = visual_data[
            visual_data["player"] == statsbomb_name
        ].copy()

        fig = plot_player_shot_map(player_events)

        if fig is None:
            return ui.p("No shot data available for this player.")

        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:900px; height:auto;">'
            f'</div>'
        )

    @render.ui
    def match_header():
        match = selected_match()

        if match is None:
            return ui.div()

        if match["Final"] == "Penalties":
            final_text = (
                f"Penalties ({int(match['home_pen_goals'])} - "
                f"{int(match['away_pen_goals'])})"
            )
        else:
            final_text = match["Final"]

        match_id = match["match_id"]
        match_events = visual_data[
            (visual_data["match_id"] == match_id) &
            (visual_data["period"] != 5)
        ]

        # Regular goals
        goals = match_events[
            (match_events["type"] == "Shot") &
            (match_events["shot_outcome"] == "Goal")
        ]

        # Own goals: use the team that benefited from the own goal
        own_goals = match_events[
            match_events["type"] == "Own Goal Against"
        ]

        home_scorers = []
        away_scorers = []

        for _, goal in goals.iterrows():
            scorer = goal["player"]
            minute = int(goal["minute"])
            team = goal["team"]

            entry = f"{scorer} {minute}'"

            if team == match["home_team"]:
                home_scorers.append((minute, entry))
            elif team == match["away_team"]:
                away_scorers.append((minute, entry))

        for _, goal in own_goals.iterrows():
            scorer = goal["player"]
            minute = int(goal["minute"])
            scoring_team = goal["team"]

            entry = f"{scorer} (OG) {minute}'"

            if scoring_team == match["home_team"]:
                away_scorers.append((minute, entry))
            elif scoring_team == match["away_team"]:
                home_scorers.append((minute, entry))

        home_scorers.sort(key=lambda x: x[0])
        away_scorers.sort(key=lambda x: x[0])

        home_scorer_text = "\n".join(
            entry for _, entry in home_scorers
        ) or " "

        away_scorer_text = "\n".join(
            entry for _, entry in away_scorers
        ) or " "


        return ui.div(

            ui.p(
                    match["competition_stage"].upper(),
                    style="""
                        font-size: 16px;
                        font-weight: 700;
                        letter-spacing: 1.5px;
                        color: #56042C;
                        margin-bottom: 18px;
                    """
                ),

            
            ui.div(
                ui.div(
                    match["home_team"],
                    style="font-size: 45px; font-weight: 600; text-align: right; padding-right: 25px;"
                ),
                ui.div(
                    f"{match['home_score']} - {match['away_score']}",
                    style="font-size: 45px; font-weight: 700; text-align: center;"
                ),
                ui.div(
                    match["away_team"],
                    style="font-size: 45px; font-weight: 600; text-align: left; padding-left: 25px;"
                ),
                style="display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;"
            ),
            ui.p(
                final_text,
                style="font-size: 25px; font-weight: 700; margin-top: 8px;"
            ),
            ui.p(
                f"{match['match_date']} • {match['stadium']}",
                style="font-size: 14px; color: #666; margin-top: 6px;font-weight: 700;"
            ),

            ui.div(
                ui.div(
                    home_scorer_text,
                    style="font-size: 14px; text-align: right; padding-right: 25px; white-space: pre-line;"
                ),
                ui.div(""),
                ui.div(
                    away_scorer_text,
                    style="font-size: 14px; text-align: left; padding-left: 25px; white-space: pre-line;"
                ),
                style="display: grid; grid-template-columns: 1fr auto 1fr; align-items: start;"
            ),

            style="""
                max-width: 1300px;
                width: 100%;
                box-sizing: border-box;
                margin: 30px auto;
                padding: 40px 50px;
                text-align: center;
                border: 4px solid #56042C;
                border-radius: 16px;
            """
        )
    
    @render.ui
    def player_header():
        player = selected_player()

        if player is None:
            return ui.div()

        country = player['team']

        team_color = team_cols.loc[
            team_cols['Country'] == country,
            'primary_color_hex'
        ].iloc[0]

        return ui.div(

            ui.p(
                player['player_name'],
                style="""
                    font-size: 45px;
                    font-weight: 700;
                    margin-bottom: 15px;
                """
            ),

            ui.p(
                {
                    'FW': 'Forward',
                    'MF': 'Midfielder',
                    'DF': 'Defender',
                    'GK': 'Goalkeeper'
                }.get(player['position'], player['position']),
                style="font-size: 24px; font-weight: 700; margin-bottom: 10px;"
            ),
            ui.p(
                f"{country} • #{player['jersey_number']}",
                style="font-size: 18px; color: #666; font-weight: 600;"
            ),

            style=f"""
                max-width: 1300px;
                width: 100%;
                box-sizing: border-box;
                margin: 30px auto;
                padding: 40px 50px;
                text-align: center;
                border: 4px solid {team_color};
                border-radius: 16px;
            """
        )

    @render.ui
    def player_summary():
        summary = player_tournament_summary()

        if summary is None:
            return ui.div()

        metrics = [
            ('Minutes', summary['minutes']),
            ('Appearances', summary['appearances']),
            ('Starts', summary['starts'])
        ]

        return ui.div(
            *[
                ui.div(
                    ui.p(
                        f"{value:,.0f}",
                        style="""
                            font-size: 36px;
                            font-weight: 700;
                            margin-bottom: 5px;
                        """
                    ),
                    ui.p(
                        label,
                        style="""
                            font-size: 16px;
                            color: #666;
                            font-weight: 600;
                        """
                    ),
                    style="text-align: center; flex: 1;"
                )
                for label, value in metrics
            ],
            style="""
                max-width: 900px;
                width: 100%;
                margin: 0 auto 35px auto;
                display: flex;
                justify-content: space-around;
                align-items: center;
            """
        )

    @render.ui
    def similar_performances():
        player = selected_player()

        if player is None:
            return ui.p("Select a player to view similar performances.")

        statsbomb_name = player["statsbomb_player"]

        similar = similar_performances_df[
            similar_performances_df["Player"] == statsbomb_name
        ].copy()

        if similar.empty:
            return ui.p(
                "Not enough tournament minutes to generate similar performances."
            )

        similar = similar.sort_values("Distance").head(3)

        cards = []

        for _, row in similar.iterrows():
            cards.append(
                ui.card(
                    ui.card_header(row["Similar Player"]),
                    ui.p(f"Team: {row['Team']}"),
                    ui.p(f"Minutes: {row['Minutes']:.0f}"),
                    ui.p(f"Similarity Index: {row['Similarity Score']:.1f}"),
                    ui.p(f"Distance: {row['Distance']:.2f}"),
                    padding=3
                )
            )

        return ui.div(
                    ui.div(
                        *cards,
                        style="""
                            display: grid;
                            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                            gap: 15px;
                        """
                    ),
                    ui.p(
                        "Note: Similarity is based on statistical performances "
                        "during the 2022 FIFA World Cup, not playing style or "
                        "player archetype.",
                        style="""
                            color: #777777;
                            font-size: 13px;
                            text-align: center;
                            margin-top: 15px;
                        """
                    )
                )

    @render.ui
    def player_pass_map():
        player = selected_player()

        if player is None:
            return ui.p("Select a player to view their pass map.")

        statsbomb_name = player["statsbomb_player"]

        team_color = team_colors.get(
            player["team"],
            "#56042C"
        )

        player_events = visual_data[
            visual_data["player"] == statsbomb_name
        ].copy()

        fig = plot_player_pass_map(
            player_events,
            "#C62828"
        )

        if fig is None:
            return ui.p("No pass data available for this player.")

        image_uri = fig_to_data_uri(fig)
        plt.close(fig)

        return ui.HTML(
            f'<div style="width:100%; text-align:center;">'
            f'<img src="{image_uri}" '
            f'style="width:100%; max-width:900px; height:auto;">'
            f'</div>'
        )

    @render.ui
    def player_category_selector():
        player = selected_player()

        if player is None or player['position'] == 'GK':
            return ui.div()

        return ui.div(
            ui.input_radio_buttons(
                "player_tournament_type",
                "Stat Category",
                choices={
                    "Standard": "Standard",
                    "Attacking": "Attacking",
                    "Midfield": "Possession & Distribution",
                    "Defending": "Defending"
                },
                selected="Standard",
                inline=True
            ),
            class_="player-stats-selector"
        )

app = App(app_ui, server)


