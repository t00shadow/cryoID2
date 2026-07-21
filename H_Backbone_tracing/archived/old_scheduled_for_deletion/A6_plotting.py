# https://python-graph-gallery.com/web-scatterplot-astronaut/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from flexitext import flexitext
from matplotlib.patches import FancyArrowPatch

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-14/astronauts.csv"
data = pd.read_csv(url)
print(data.head())

data["age_at_mission"] = data["year_of_mission"] - data["year_of_birth"]

year_of_mission = data["year_of_mission"].values
age_at_mission = data["age_at_mission"].values
sex = data["sex"].values
hours_mission = data["hours_mission"].values
total_number_of_missions = data["total_number_of_missions"].values

DARKBLUE = "#1d1330"
ORANGE = "#ffa72b"
LIGHTBLUE = "#e1f7fa"

def scale(x, lower=0, upper=1):
    return ((x - np.min(x)) / (np.max(x) - np.min(x))) * (upper - lower)  + lower

alpha = scale(total_number_of_missions, 0.1, 1)
sizes = scale(hours_mission, 10, 200)

# Also create an array with the colors
colors = np.where(sex == "male", LIGHTBLUE, ORANGE)

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(age_at_mission, year_of_mission)

plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

# Set background color
fig.set_facecolor(DARKBLUE)
ax.set_facecolor(DARKBLUE)

ax.scatter(age_at_mission, year_of_mission, c=colors, s=sizes, alpha=alpha)

plt.show()
