"""
Build a single unified dataset from the 4 HR data sources.

Joins:
  - general_data.csv          (base table, 24 cols)
  - employee_survey_data.csv  (3 cols)   → LEFT JOIN on EmployeeID
  - manager_survey_data.csv   (2 cols)   → LEFT JOIN on EmployeeID
  - in_out_time/              (6 cols)   → LEFT JOIN on EmployeeID (row index)

Output: data/dataset_final.csv
"""

import pandas as pd


def load_general_data(path="data/general_data.csv"):
    return pd.read_csv(path)


def load_employee_survey(path="data/employee_survey_data.csv"):
    return pd.read_csv(path)


def load_manager_survey(path="data/manager_survey_data.csv"):
    return pd.read_csv(path)


def load_time_features(in_path="data/in_out_time/in_time.csv",
                       out_path="data/in_out_time/out_time.csv"):
    df_in = pd.read_csv(in_path, index_col=0)
    df_out = pd.read_csv(out_path, index_col=0)

    df_in_dt = df_in.apply(pd.to_datetime, errors="coerce")
    df_out_dt = df_out.apply(pd.to_datetime, errors="coerce")

    df_duration = (df_out_dt - df_in_dt).apply(
        lambda col: col.dt.total_seconds() / 3600
    )
    df_in_hours = df_in_dt.apply(lambda col: col.dt.hour + col.dt.minute / 60)
    df_out_hours = df_out_dt.apply(lambda col: col.dt.hour + col.dt.minute / 60)

    return pd.DataFrame({
        "EmployeeID": df_in.index,
        "avg_in_hour": df_in_hours.mean(axis=1).values,
        "avg_out_hour": df_out_hours.mean(axis=1).values,
        "avg_work_hours": df_duration.mean(axis=1).values,
        "std_work_hours": df_duration.std(axis=1).values,
        "nb_days_present": df_duration.notna().sum(axis=1).values,
        "nb_days_absent": df_duration.isna().sum(axis=1).values,
    })


def build_dataset():
    df = load_general_data()
    df = df.merge(load_employee_survey(), on="EmployeeID", how="left")
    df = df.merge(load_manager_survey(), on="EmployeeID", how="left")
    df = df.merge(load_time_features(), on="EmployeeID", how="left")
    return df


if __name__ == "__main__":
    df = build_dataset()
    output_path = "data/dataset_final.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
