std_msgs/Header header
uint32 height
uint32 width

# Intrinsic camera matrix K (flattened 3 x 3 row-major)
# [fx, 0, cx, 0, fy, cy, 0, 0, 1]
float64[9] k

# Metric depth
float32[] depth_data

float32 min_depth
float32 max_depth
