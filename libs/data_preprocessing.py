from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def preprocess_data(df, strategy="most_frequent"):

    # df.dropna(inplace=True)
    x_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy=strategy)
    new_x = imputer.fit_transform(x_copy)

    y_data = df.iloc[:, -1].values
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y_data)

    return new_x, y


def split_data(X, y, test_size=0.15, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
