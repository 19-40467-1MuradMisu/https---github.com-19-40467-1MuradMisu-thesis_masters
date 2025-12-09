import numpy as np
import pandas as pd


# =====================================================================
# INTERPOLATION FUNCTION (SIMPLE + FULLY COMMENTED)
# =====================================================================
def interpolate_with_segments(s_orig, x_orig, y_orig, s_grid):
    """
    Linearly interpolate (x, y) coordinates along a cumulative distance axis.

    This function answers the question:
        “If the path has points at distances s_orig,
         what are the coordinates at distances s_grid?”

    HOW IT WORKS:
    -------------
    - The original path is made of segments:
            P0 ---- P1 ---- P2 ---- ... ---- Pn
      where each Pi has cumulative distance s_orig[i].

    - For each target distance g in s_grid:
        1) Find segment [j, j+1] where s_orig[j] <= g <= s_orig[j+1]
        2) Compute how far between these two points the value g lies (a fraction t)
        3) Use linear interpolation to find x,y at that exact location.
    """

    n = len(s_orig)  # number of original points

    # Prepare arrays to store new interpolated results
    x_new = np.zeros_like(s_grid, dtype=float)
    y_new = np.zeros_like(s_grid, dtype=float)

    # Track which original segment created each grid point
    seg_idx = np.zeros_like(s_grid, dtype=int)

    j = 0  # start assuming segment [0,1]

    # Loop over each desired distance value g
    for k, g in enumerate(s_grid):

        # ------------------------------------------------------------
        # STEP 1: Find which segment contains this distance g
        # ------------------------------------------------------------
        # Move j forward while g is still beyond s_orig[j+1]
        # This guarantees eventually:
        #      s_orig[j] <= g <= s_orig[j+1]
        while j < n - 2 and g > s_orig[j + 1]:
            j += 1

        # Segment boundaries in "distance space"
        s_i = s_orig[j]
        s_ip1 = s_orig[j + 1]

        # ------------------------------------------------------------
        # STEP 2: Compute interpolation fraction
        # ------------------------------------------------------------
        denom = s_ip1 - s_i
        if denom == 0:
            # Rare case: two identical s_orig points
            t = 0.0
        else:
            # Fraction of distance between the endpoints
            t = (g - s_i) / denom

        # ------------------------------------------------------------
        # STEP 3: Get end-point coordinates of the current segment
        # ------------------------------------------------------------
        x_i = x_orig[j]
        x_ip1 = x_orig[j + 1]
        y_i = y_orig[j]
        y_ip1 = y_orig[j + 1]

        # ------------------------------------------------------------
        # STEP 4: Linear interpolation in XY space
        # ------------------------------------------------------------
        x_new[k] = x_i + t * (x_ip1 - x_i)
        y_new[k] = y_i + t * (y_ip1 - y_i)

        # Store which segment was used (for debugging if needed)
        seg_idx[k] = j

    return x_new, y_new, seg_idx



# =====================================================================
# MAIN DEVIATION FUNCTION USING YOUR FULL ALGORITHM
# =====================================================================
def compute_deviation_with_overlap(
    lane_x,
    lane_y,
    veh_x,
    veh_y,
    step=0.25
):
    """
    Compute lane deviation using your EXACT algorithm:

    ALGORITHM STEPS
    ---------------
    1. Compute cumulative distance of raw lane and vehicle.
    2. Interpolate BOTH trajectories independently at 0.25m spacing.
    3. Choose the SHORTER interpolated trajectory.
    4. Use the shorter one's:
           - First point  → nearest point on the longer → start align index
           - Last point   → nearest point on the longer → end align index
    5. Crop both trajectories to the common overlapping region.
    6. Recompute cumulative distances for both CROPPED paths.
    7. Create a COMMON s_grid using minimum length of both cropped paths.
    8. Re-interpolate BOTH cropped paths onto that same s_grid.
    9. Compute deviation = Euclidean distance point-to-point.
    """

    # ------------------------------------------------------------
    # STEP 1: Compute cumulative distance for RAW inputs
    # ------------------------------------------------------------
    def cumulative_distance(x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx*dx + dy*dy)
        return np.insert(np.cumsum(seg_len), 0, 0.0)

    s_lane_raw = cumulative_distance(lane_x, lane_y)
    s_veh_raw  = cumulative_distance(veh_x, veh_y)

    L_lane_raw = s_lane_raw[-1]
    L_veh_raw  = s_veh_raw[-1]

    # ------------------------------------------------------------
    # STEP 2: Interpolate BOTH paths independently at step=0.25m
    # ------------------------------------------------------------
    s_lane_grid = np.arange(0.0, L_lane_raw + 1e-9, step)
    s_veh_grid  = np.arange(0.0, L_veh_raw + 1e-9, step)

    lane_x_i, lane_y_i, _ = interpolate_with_segments(
        s_lane_raw, lane_x, lane_y, s_lane_grid
    )
    veh_x_i, veh_y_i, _ = interpolate_with_segments(
        s_veh_raw, veh_x, veh_y, s_veh_grid
    )

    # ------------------------------------------------------------
    # STEP 3: Decide which interpolated path is shorter
    # ------------------------------------------------------------
    if L_lane_raw <= L_veh_raw:
        short_x, short_y, short_s = lane_x_i, lane_y_i, s_lane_grid
        long_x,  long_y,  long_s  = veh_x_i, veh_y_i, s_veh_grid
        short_name = "lane"
    else:
        short_x, short_y, short_s = veh_x_i, veh_y_i, s_veh_grid
        long_x,  long_y,  long_s  = lane_x_i, lane_y_i, s_lane_grid
        short_name = "veh"

    # ------------------------------------------------------------
    # STEP 4: Find nearest START and END positions on the longer path
    # ------------------------------------------------------------
    def find_nearest_index(x_arr, y_arr, px, py):
        dx = x_arr - px
        dy = y_arr - py
        dist2 = dx*dx + dy*dy
        idx = np.argmin(dist2)
        return idx

    # First point of shorter path
    start_idx_long = find_nearest_index(long_x, long_y, short_x[0], short_y[0])

    # Last point of shorter path
    end_idx_long = find_nearest_index(long_x, long_y, short_x[-1], short_y[-1])

    # Ensure ordering
    if end_idx_long < start_idx_long:
        start_idx_long, end_idx_long = end_idx_long, start_idx_long

    # ------------------------------------------------------------
    # STEP 5: Crop both paths to the overlapping region
    # ------------------------------------------------------------
    # Shorter path → keep whole thing
    short_x_crop = short_x
    short_y_crop = short_y

    # Longer path → keep only matched segment
    long_x_crop  = long_x[start_idx_long : end_idx_long + 1]
    long_y_crop  = long_y[start_idx_long : end_idx_long + 1]

    # ------------------------------------------------------------
    # STEP 6: Recompute s-values for the cropped paths
    # ------------------------------------------------------------
    s_short_crop = cumulative_distance(short_x_crop, short_y_crop)
    s_long_crop  = cumulative_distance(long_x_crop,  long_y_crop)

    L_short_crop = s_short_crop[-1]
    L_long_crop  = s_long_crop[-1]

    # The usable overlapping length is the minimum of the two
    L_common = min(L_short_crop, L_long_crop)

    # ------------------------------------------------------------
    # STEP 7: Build final common grid
    # ------------------------------------------------------------
    s_grid = np.arange(0.0, L_common + 1e-9, step)

    # ------------------------------------------------------------
    # STEP 8: Final interpolation of both cropped paths
    # ------------------------------------------------------------
    short_x_final, short_y_final, _ = interpolate_with_segments(
        s_short_crop, short_x_crop, short_y_crop, s_grid
    )
    long_x_final, long_y_final, _ = interpolate_with_segments(
        s_long_crop, long_x_crop, long_y_crop, s_grid
    )

    # ------------------------------------------------------------
    # STEP 9: Map short/long back to lane/vehicle
    # ------------------------------------------------------------
    if short_name == "lane":
        lane_x_f, lane_y_f = short_x_final, short_y_final
        veh_x_f,  veh_y_f  = long_x_final,  long_y_final
    else:
        veh_x_f,  veh_y_f  = short_x_final, short_y_final
        lane_x_f, lane_y_f = long_x_final,  long_y_final

    # ------------------------------------------------------------
    # STEP 10: Compute deviation
    # ------------------------------------------------------------
    dx = veh_x_f - lane_x_f
    dy = veh_y_f - lane_y_f
    deviation = np.sqrt(dx*dx + dy*dy)

    return s_grid, lane_x_f, lane_y_f, veh_x_f, veh_y_f, deviation



# =====================================================================
# EXAMPLE USAGE
# =====================================================================
if __name__ == "__main__":
    lane_df = pd.read_csv("lane_csv.csv")
    veh_df  = pd.read_csv("veh_csv.csv")

    lane_x = lane_df["x"].to_numpy()
    lane_y = lane_df["y"].to_numpy()
    veh_x  = veh_df["x"].to_numpy()
    veh_y  = veh_df["y"].to_numpy()

    s_grid, lane_x_f, lane_y_f, veh_x_f, veh_y_f, dev = compute_deviation_with_overlap(
        lane_x, lane_y, veh_x, veh_y, step=0.25
    )

    result_df = pd.DataFrame({
        "s": s_grid,
        "lane_x": lane_x_f,
        "lane_y": lane_y_f,
        "veh_x": veh_x_f,
        "veh_y": veh_y_f,
        "deviation": dev
    })

    result_df.to_csv("deviation_output.csv", index=False)

    print("Mean deviation:", dev.mean())
    print("Max deviation:", dev.max())
