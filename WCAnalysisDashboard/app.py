from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

import ast
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
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
            
        """)
    ),
    ui.h1("2022 World Cup Dashboard"),

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
                            "Midfield": "Midfield",
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
                ui.output_data_frame("standard_player_table"),
                class_="player-stats-section"
            ),
            ui.div(
                ui.h2(
                    "Goalkeeper Stats",
                    style="color: #56042C; text-align: center; margin-top: 30px;"
                ),
                ui.input_select(
                    "keeper_team_filter",
                    "Filter by Team",
                    choices={"All Teams": "All Teams"},
                    selected="All Teams"
                ),
                ui.output_data_frame("keeper_player_table"),
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



    @render.data_frame
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
            return render.DataGrid(
                data.iloc[0:0][columns],
                filters=False,
                height="auto"
            )

        match_id = int(input.match())

        df = data[data["game_id"] == match_id].copy()
        if "xG" in df.columns:
            df["xG"] = df["xG"].round(2)

        team_filter = input.player_team_filter()

        if team_filter != "All Teams":
            df = df[df["Team"] == team_filter]

        return render.DataGrid(
            df[columns].sort_values(
                ["Team", "Position", "Player"]
            ),
            filters=False,
            height="auto"
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

    @render.data_frame
    def keeper_player_table():
        columns = [
            "Player", "Position", "Minutes", "Team",
            "Shots Faced", "Saves", "Goals Conceded",
            "Passes", "Clearances"
        ]

        match = selected_match()

        if match is None:
            return render.DataGrid(
                keeper_stats.iloc[0:0][columns],
                filters=False,
                height="auto"
            )

        match_id = int(input.match())

        df = keeper_stats[
            keeper_stats["game_id"] == match_id
        ].copy()

        team_filter = input.keeper_team_filter()

        if team_filter != "All Teams":
            df = df[df["Team"] == team_filter]

        return render.DataGrid(
            df[columns].sort_values(
                ["Team", "Player"]
            ),
            filters=False,
            height="auto"
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

app = App(app_ui, server)

