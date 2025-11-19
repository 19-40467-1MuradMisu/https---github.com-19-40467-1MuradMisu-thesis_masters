import numpy as np
import pandas as pd


def interpolate_with_segments(s_orig, x_orig, y_orig, s_grid):
    """
    Manual linear interpolation on a cumulative distance axis.

    Parameters:
        s_orig : 1D array
            Original cumulative distance values of the trajectory
            (e.g., [0.0, 0.7, 1.5, ...]).
        x_orig, y_orig : 1D arrays
            Original x and y coordinates corresponding to s_orig.
        s_grid : 1D array
            Target distance grid where we want interpolated points
            (e.g., [0.0, 0.25, 0.5, 0.75, ...]).

    For each grid point g in s_grid, we find the segment index i such that:
        s_i <= g <= s_{i+1}

    Then compute the interpolation fraction:
        t = (g - s_i) / (s_{i+1} - s_i)

    And use that to compute the interpolated coordinates:
        x_new = x_i + t * (x_{i+1} - x_i)
        y_new = y_i + t * (y_{i+1} - y_i)

    Returns:
        x_new : 1D array
            Interpolated x-coordinates on s_grid.
        y_new : 1D array
            Interpolated y-coordinates on s_grid.
        seg_idx : 1D int array
            For each grid point, the index i of the original segment
            [i, i+1] that was used for interpolation.
    """
    n = len(s_orig)

    # Arrays to hold interpolated coordinates and segment indexces
    x_new = np.zeros_like(s_grid, dtype=float)
    y_new = np.zeros_like(s_grid, dtype=float)
    seg_idx = np.zeros_like(s_grid, dtype=int)

    j = 0  # current segment index in the original trajectory

    # Loop over all grid points g
    for k, g in enumerate(s_grid):
        # Move j forward until we find the segment [s_j, s_{j+1}] that contains g
        # Condition: s_j <= g <= s_{j+1}
        # We increase j while g is larger than the upper bound s_{j+1}
        while j < n - 2 and g > s_orig[j + 1]:
            j += 1

        # Segment boundaries in the original cumulative distance array
        s_i = s_orig[j]
        s_ip1 = s_orig[j + 1]

        # Avoid division by zero in case of a zero-length segment
        denom = s_ip1 - s_i
        if denom == 0:
            t = 0.0
        else:
            # Interpolation fraction along the segment [s_i, s_{i+1}]
            t = (g - s_i) / denom

        # Original coordinates at the segment endpoints
        x_i = x_orig[j]
        x_ip1 = x_orig[j + 1]
        y_i = y_orig[j]
        y_ip1 = y_orig[j + 1]

        # Linear interpolation for x and y
        x_new[k] = x_i + t * (x_ip1 - x_i)
        y_new[k] = y_i + t * (y_ip1 - y_i)

        # Record which original segment index was used
        seg_idx[k] = j

    return x_new, y_new, seg_idx


def compute_deviation_from_csv_simple(
    lane_csv,
    veh_csv,
    x_col="x",
    y_col="y",
    step=0.25,
    out_csv=None
):
    """
    Compute lateral deviation between a lane trajectory and a vehicle trajectory.

    Inputs:
        lane_csv : str
            Path to CSV file containing lane centerline coordinates.
        veh_csv : str
            Path to CSV file containing vehicle trajectory coordinates.
        x_col : str
            Name of the x-column in both CSVs (e.g. "x").
        y_col : str
            Name of the y-column in both CSVs (e.g. "y").
        step : float
            Distance step for the uniform grid (e.g., 0.25 m).
        out_csv : str or None
            Optional path to save the resulting deviation data as CSV.

    Method (always the same, regardless of original point counts):
        1. Read (x, y) from both CSVs.
        2. Compute cumulative distance for both trajectories (s_lane, s_veh).
        3. Determine the common length:
               common_L = min(s_lane[-1], s_veh[-1])
        4. Create a uniform distance grid from 0 to common_L with spacing = step:
               s_grid = [0, step, 2*step, ..., common_L]
        5. For each grid point, perform manual linear interpolation on:
               (s_lane, lane_x, lane_y) and (s_veh, veh_x, veh_y)
           using:
               x_new = x_i + (g - s_i)/(s_{i+1} - s_i) * (x_{i+1} - x_i)
        6. Compute point-wise Euclidean deviation between the two
           interpolated trajectories.
        7. Return a DataFrame with:
               s, lane_x, lane_y, veh_x, veh_y, deviation,
               lane_seg_idx, veh_seg_idx

    This ensures both trajectories are compared on the same physical distance axis
    and the same grid, even if original sampling rates or point counts differ.
    """

    # -----------------------------
    # Step 1: read CSVs
    # -----------------------------
    # Read lane and vehicle trajectories from CSV files
    lane_df = pd.read_csv(lane_csv)
    veh_df  = pd.read_csv(veh_csv)

    # Extract x and y columns as NumPy arrays
    lane_x = lane_df[x_col].to_numpy()
    lane_y = lane_df[y_col].to_numpy()

    veh_x  = veh_df[x_col].to_numpy()
    veh_y  = veh_df[y_col].to_numpy()

    # Store number of points for basic info
    n_lane = len(lane_x)
    n_veh  = len(veh_x)

    print(f"[INFO] Lane points: {n_lane}, Vehicle points: {n_veh}")

    # We need at least 2 points in each trajectory to build segments
    if n_lane < 2 or n_veh < 2:
        raise ValueError("Both trajectories must have at least 2 points for interpolation.")

    # -----------------------------
    # Step 2: cumulative distances
    # -----------------------------
    # Compute segment-wise differences for lane: dx, dy between consecutive points
    dx_lane = np.diff(lane_x)
    dy_lane = np.diff(lane_y)
    # Segment lengths for lane
    lane_seg_len = np.sqrt(dx_lane**2 + dy_lane**2) # (Local distances)Euclidean distance between consecutive points
    # Cumulative distance along lane, starting from 0.0
    s_lane = np.insert(np.cumsum(lane_seg_len), 0, 0.0) # (Global Distance axis) cumulative sum of segment lengths

    # Same process for vehicle trajectory
    dx_veh = np.diff(veh_x)
    dy_veh = np.diff(veh_y)
    veh_seg_len = np.sqrt(dx_veh**2 + dy_veh**2) 
    s_veh = np.insert(np.cumsum(veh_seg_len), 0, 0.0)

    # -----------------------------
    # Step 3: common length & grid
    # -----------------------------
    # Use only the overlapping distance range of the two trajectories (taking the minimum length)
    common_L = min(s_lane[-1], s_veh[-1])
    print(f"[INFO] Common length used: {common_L:.2f} m")

    # Build a uniform grid of distances from 0 to common_L
    # with spacing = step (e.g., 0.25 m)
    s_grid = np.arange(0.0, common_L + 1e-9, step)  # 0, step, 2*step, ...

    # -----------------------------
    # Step 4: interpolate both trajectories on the same grid
    # -----------------------------
    # Interpolate lane trajectory onto s_grid
    lane_x_used, lane_y_used, lane_seg_idx = interpolate_with_segments(
        s_lane, lane_x, lane_y, s_grid
    )

    # Interpolate vehicle trajectory onto s_grid
    veh_x_used, veh_y_used, veh_seg_idx = interpolate_with_segments(
        s_veh, veh_x, veh_y, s_grid
    )

    # -----------------------------
    # Step 5: Euclidean deviation
    # -----------------------------
    # Compute point-wise difference in x and y
    dx = veh_x_used - lane_x_used
    dy = veh_y_used - lane_y_used

    # Euclidean distance between lane and vehicle at each grid point
    deviation = np.sqrt(dx**2 + dy**2)

    # -----------------------------
    # Step 6: assemble DataFrame
    # -----------------------------
    # Build a DataFrame with all relevant information
    data = {
        "s": s_grid,              # distance from start (common grid)
        "lane_x": lane_x_used,    # interpolated lane x
        "lane_y": lane_y_used,    # interpolated lane y
        "veh_x":  veh_x_used,     # interpolated vehicle x
        "veh_y":  veh_y_used,     # interpolated vehicle y
        "deviation": deviation,   # Euclidean deviation at each grid point
        # segment indices: which original segment [i, i+1]
        # was used to interpolate each grid point
        "lane_seg_idx": lane_seg_idx,
        "veh_seg_idx":  veh_seg_idx,
    }
    df_dev = pd.DataFrame(data)

    # optional: save to CSV
    if out_csv is not None:
        df_dev.to_csv(out_csv, index=False)
        print(f"[INFO] Deviation data saved to: {out_csv}")

    # Print some summary statistics
    print(f"[INFO] Mean deviation: {deviation.mean():.3f} m")
    print(f"[INFO] RMS deviation:  {np.sqrt((deviation**2).mean()):.3f} m")

    return df_dev


# Example usage
if __name__ == "__main__":
    # Input CSV paths for lane and vehicle trajectories
    lane_file = "lane_csv.csv"
    veh_file  = "veh_csv.csv"

    # Run deviation computation on these files
    df_result = compute_deviation_from_csv_simple(
        lane_csv=lane_file,
        veh_csv=veh_file,
        x_col="x",   # column names in your CSV files
        y_col="y",
        step=0.25,   # grid spacing in meters
        out_csv="deviation_output.csv"
    )
