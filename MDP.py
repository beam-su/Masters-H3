import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import h3
from shapely.geometry import Point
from mdptoolbox.mdp import ValueIteration

# Load the hexagonal geojson file
hex_gdf = gpd.read_file("geojson\\hex_dataset.geojson")

# Ensure necessary columns exist
if "avg_imd_score" not in hex_gdf.columns or "total_population" not in hex_gdf.columns:
    raise ValueError("GeoJSON file must contain 'avg_imd_score' and 'total_population' columns.")

# Create an adjacency graph based on H3 neighbors
G = nx.Graph()
hex_indices = hex_gdf["h3_index"]

for h in hex_indices:
    neighbors = h3.grid_disk(h, 1)  # Get neighboring hexagons
    for n in neighbors:
        if n in hex_indices.values:
            G.add_edge(h, n)

# Define MDP Components
num_states = len(hex_indices)
states = list(hex_indices)
rewards = np.zeros(num_states)

# Define reward function: Higher population + lower IMD score = better station location
for i, h in enumerate(states):
    row = hex_gdf[hex_gdf["h3_index"] == h]
    if not row.empty:
        imd_score = row["avg_imd_score"].values[0]
        population = row["total_population"].values[0]
        rewards[i] = population / (imd_score + 1e-6)  # Avoid division by zero

# Normalize rewards
rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

# Define the number of actions
num_actions = 2  # Action 0: Do nothing, Action 1: Place a station

# Initialize a 3D transition matrix
transition_matrix = np.zeros((num_actions, num_states, num_states))

# Define transitions for Action 0 (Do nothing)
for i in range(num_states):
    transition_matrix[0, i, i] = 1.0  # Stay in the same state

# Define transitions for Action 1 (Place a station)
for i, h in enumerate(states):
    neighbors = h3.grid_disk(h, 1)
    valid_neighbors = [n for n in neighbors if n in states]
    num_neighbors = len(valid_neighbors)
    
    if num_neighbors > 0:
        prob = 1 / num_neighbors
        for n in valid_neighbors:
            j = states.index(n)
            transition_matrix[1, i, j] = prob
    transition_matrix[1, i, i] += 1.0  # Add self-loop probability

# Normalize the transition matrix to ensure it is stochastic
for a in range(num_actions):
    for s in range(num_states):
        row_sum = np.sum(transition_matrix[a, s, :])
        if row_sum > 0:
            transition_matrix[a, s, :] /= row_sum  # Normalize row to sum to 1

# Create an MDP using Value Iteration
mdp = ValueIteration(transition_matrix, rewards, discount=0.9)
mdp.run()

# Load the existing bike stations GeoJSON file
existing_stations_gdf = gpd.read_file("geojson\\existing_bike_stations.geojson")

# Ensure the CRS (Coordinate Reference System) matches between datasets
if hex_gdf.crs != existing_stations_gdf.crs:
    existing_stations_gdf = existing_stations_gdf.to_crs(hex_gdf.crs)

# Extract best 5 hexagons
best_hex_indices = np.argsort(mdp.V)[-5:]  # Top 5 states with highest value function
best_hexes = [states[i] for i in best_hex_indices]

# Extract final recommendations
best_locations = hex_gdf[hex_gdf["h3_index"].isin(best_hexes)][["h3_index", "geometry", "total_population", "avg_imd_score"]]

# Filter recommendations to ensure they are within 800 meters of existing bike stations
best_locations = best_locations[
    best_locations["geometry"].apply(
        lambda geom: existing_stations_gdf.distance(geom).min() <= 800
    )
]

print(best_locations)

# Save results to file
best_locations.to_file("best_hexagons.geojson", driver="GeoJSON")
best_locations.drop(columns="geometry").to_csv("best_hexagons.csv", index=False)
print("Results saved to 'best_hexagons.geojson' and 'best_hexagons.csv")