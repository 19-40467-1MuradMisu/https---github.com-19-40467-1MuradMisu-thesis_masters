import numpy as np
import pandas as pd


def compute_deviation_from_csv_simple(
    lane_csv,
    veh_csv,
    x_col="x",
    y_col="y",
    step=0.25,
    out_csv=None
):
    """
    2 ta CSV input ney (lane & vehicle),
    jodi point count same hoy: direct Euclidean distance diye deviation,
    jodi alada hoy: cumulative distance + 0.25m grid + interpolation + deviation.
    """

    # ---------------------------------
    # Step 1: CSV theke x, y read kora
    # ---------------------------------
    lane_df = pd.read_csv(lane_csv)
    veh_df  = pd.read_csv(veh_csv)

    lane_x = lane_df[x_col].to_numpy()
    lane_y = lane_df[y_col].to_numpy()

    veh_x  = veh_df[x_col].to_numpy()
    veh_y  = veh_df[y_col].to_numpy()

    n_lane = len(lane_x)
    n_veh  = len(veh_x)

    print(f"[INFO] Lane points: {n_lane}, Vehicle points: {n_veh}")

    # ---------------------------------
    # Step 2: point count compare
    # ---------------------------------
    if n_lane == n_veh:
        print("[INFO] Same number of points → no interpolation, direct comparison.")

        # chaile distance axis hishebe lane cumulative distance use kora jay
        dx_seg = np.diff(lane_x)
        dy_seg = np.diff(lane_y)
        seg_len = np.sqrt(dx_seg**2 + dy_seg**2)
        s_grid = np.insert(np.cumsum(seg_len), 0, 0.0)   # distance from start

        lane_x_used, lane_y_used = lane_x, lane_y
        veh_x_used,  veh_y_used  = veh_x,  veh_y

    else:
        print("[INFO] Different number of points → interpolation mode.")

        # ---------------------------------
        # Step 3: lane cumulative distance
        # ---------------------------------
        dx_lane = np.diff(lane_x)
        dy_lane = np.diff(lane_y)
        lane_seg_len = np.sqrt(dx_lane**2 + dy_lane**2)
        s_lane = np.insert(np.cumsum(lane_seg_len), 0, 0.0)

        # ---------------------------------
        # Step 4: vehicle cumulative distance
        # ---------------------------------
        dx_veh = np.diff(veh_x)
        dy_veh = np.diff(veh_y)
        veh_seg_len = np.sqrt(dx_veh**2 + dy_veh**2)
        s_veh = np.insert(np.cumsum(veh_seg_len), 0, 0.0)

        # ---------------------------------
        # Step 5: common length & 0.25 m grid
        # ---------------------------------
        common_L = min(s_lane[-1], s_veh[-1])
        print(f"[INFO] Common length used: {common_L:.2f} m")

        s_grid = np.arange(0.0, common_L + 1e-9, step)  # 0, 0.25, 0.5, ...

        # ---------------------------------
        # Step 6: duita trajectory-kei grid e interpolate
        # ---------------------------------
        lane_x_used = np.interp(s_grid, s_lane, lane_x)
        lane_y_used = np.interp(s_grid, s_lane, lane_y)

        veh_x_used  = np.interp(s_grid, s_veh, veh_x)
        veh_y_used  = np.interp(s_grid, s_veh, veh_y)

    # ---------------------------------
    # Step 7: pair-wise Euclidean distance = deviation
    # ---------------------------------
    dx = veh_x_used - lane_x_used
    dy = veh_y_used - lane_y_used
    deviation = np.sqrt(dx**2 + dy**2)

    # ---------------------------------
    # Step 8: DataFrame ready kora (export er jonno)
    # ---------------------------------
    data = {
        "s": s_grid,              # distance axis / index axis
        "lane_x": lane_x_used,
        "lane_y": lane_y_used,
        "veh_x":  veh_x_used,
        "veh_y":  veh_y_used,
        "deviation": deviation
    }
    df_dev = pd.DataFrame(data)

    # optional: CSV te save
    if out_csv is not None:
        df_dev.to_csv(out_csv, index=False)
        print(f"[INFO] Deviation data saved to: {out_csv}")

    print(f"[INFO] Mean deviation: {deviation.mean():.3f} m")
    print(f"[INFO] RMS deviation:  {np.sqrt((deviation**2).mean()):.3f} m")

    return df_dev


# Example use
if __name__ == "__main__":
    lane_file = "lane_csv.csv"
    veh_file  = "veh_csv.csv"

    df_result = compute_deviation_from_csv_simple(
        lane_csv=lane_file,
        veh_csv=veh_file,
        x_col="x",   # tumar csv te jei column name, oita dao
        y_col="y",
        step=0.25,
        out_csv="deviation_output.csv"
    )
