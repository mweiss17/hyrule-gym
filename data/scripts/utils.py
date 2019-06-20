
def find_nearby_nodes(coords_df, coords, radius):
    return coords_df[(coords_df.x > coords.x - radius) & (coords_df.x < coords.x + radius) & (coords_df.y > coords.y - radius) & (coords_df.y < coords.y + radius)]
