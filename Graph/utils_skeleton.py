CAR_KEYPOINTS_24 = [
    'front_up_right',       # 1
    'front_up_left',        # 2
    'front_light_right',    # 3
    'front_light_left',     # 4
    'front_low_right',      # 5
    'front_low_left',       # 6
    'central_up_left',      # 7
    'front_wheel_left',     # 8
    'rear_wheel_left',      # 9
    'rear_corner_left',     # 10
    'rear_up_left',         # 11
    'rear_up_right',        # 12
    'rear_light_left',      # 13
    'rear_light_right',     # 14
    'rear_low_left',        # 15
    'rear_low_right',       # 16
    'central_up_right',     # 17
    'rear_corner_right',    # 18
    'rear_wheel_right',     # 19
    'front_wheel_right',    # 20
    'rear_plate_left',      # 21
    'rear_plate_right',     # 22
    'mirror_edge_left',     # 23
    'mirror_edge_right',    # 24
]

SKELETON_ORIG_66_POINTS = [
    (49, 46), (49, 8), (49, 57), (8, 0), (8, 11), (57, 0),
    (57, 52), (0, 5), (52, 5), (5, 7),  # frontal
    (7, 20), (11, 23), (20, 23), (23, 25), (34, 32),
    (9, 11), (9, 7), (9, 20), (7, 0), (9, 0), (9, 8),  # L-lat
    (24, 33), (24, 25), (24, 11), (25, 32), (25, 28),
    (33, 32), (33, 46), (32, 29), (28, 29),  # rear
    (65, 64), (65, 25), (65, 28), (65, 20), (64, 29),
    (64, 32), (64, 37), (29, 37), (28, 20),  # new rear
    (34, 37), (34, 46), (37, 50), (50, 52), (46, 48), (48, 37),
    (48, 49), (50, 57), (48, 57), (48, 50)
]
# positions of ids are to be used
KPS_MAPPING = [49, 8, 57, 0, 52, 5, 11, 7, 20, 23, 24, 33, 25, 32, 28,
               29, 46, 34, 37, 50, 65, 64, 9, 48]

SKELETON_24_MAPPING = [(1, 17), (1, 2), (1, 3), (2, 4), (2, 7), (3, 4),
    (3, 5), (4, 6), (5, 6), (6, 8), (8, 9), (7, 10),
    (9, 10), (10, 13), (18, 14), (11, 7), (11, 8),
    (11, 9), (8, 4), (11, 4), (11, 2), (12, 19),
    (12, 13), (12, 7), (13, 14), (13, 15), (19, 14),
    (19, 17), (14, 16), (15, 16), (21, 20), (21, 13),
    (21, 15), (21, 9), (20, 16), (20, 14), (20, 18),
    (16, 18), (15, 9), (18, 19), (18, 17), (19, 3),
    (17, 24), (17, 25), (17, 19), (19, 3)]

Visible_points_sample = [1, 2, 6, 17, 24] # sample of visible points

# filtered points
filtered_edges = [(u, v) for u, v in SKELETON_24_MAPPING if u in Visible_points_sample and v in Visible_points_sample]

kps_lookup = {kp: idx + 1 for idx, kp in enumerate(KPS_MAPPING)}

# Step 2: Create the edges of the graph by mapping the keypoints in SKELETON_ORIG to their indices in KPS_MAPPING.
edges = []

for kp1, kp2 in SKELETON_ORIG_66_POINTS:
    if kp1 in kps_lookup and kp2 in kps_lookup:
        # Get the index positions for both keypoints and add the edge to the graph
        idx1 = kps_lookup[kp1]
        idx2 = kps_lookup[kp2]
        edges.append((idx1, idx2))
# Print the edges
