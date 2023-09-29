def validate_data(df):
    target_column = df.columns[-1]

    # Check for NaN values in the target column
    if df[target_column].isna().any().any():
        print("NaN value detected")
        return False

    # Check for Imbalance in Target Classes
    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()

    if min_class_count > 0:
        print("Data Validation Failed: Imbalance in Target Classes Detected")
        return False

    print("Data Validation Successful")
    return True
