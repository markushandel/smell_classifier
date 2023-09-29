def validate_data(df):
    target_column = df.columns[-1]

    # Check for NaN values in the DataFrame
    if df.isna().any().any():
        print("NaN value detected")

    # Check for Imbalance in Target Classes
    class_counts = df[target_column].value_counts()
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()

    if min_class_count / max_class_count < 0.2:
        print("Data Validation Failed: Imbalance in Target Classes Detected")
        return False

    # Check for identical rows
    if df.duplicated().any():
        print("Duplicate rows detected")

    print("Data Validation Successful")
    return True
