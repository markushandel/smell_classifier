import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
import time
from libs import load_data, preprocess_data, split_data, validate_data
from models import NaiveBayesModel, RandomForestModel, DecisionTreeModel, SVMModel, NeuralNetworkModel
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

# Map for models
models = {
    "Naive Bayes": NaiveBayesModel(),
    "Random Forest": RandomForestModel(),
    "Self-Trained NN": NeuralNetworkModel(),
    "Decision Tree": DecisionTreeModel(),
    "SVM": SVMModel(),
}


def init_sidebar():
    st.sidebar.header("Settings")
    dataset_name = st.sidebar.selectbox("Select Dataset", ["data-class", "feature-envy", "god-class", "long-method"])
    model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
    return dataset_name, model_name


@st.cache_data
def load_and_preprocess(dn):
    df = load_data(f'data/{dn}.arff')
    if validate_data(df):
        return preprocess_data(df)
    else:
        return None, None


def train(_model, x_train, y_train, x_test, y_test):
    start_time = time.time()  # Start the timer
    _model.train(x_train, y_train)
    elapsed_time = time.time() - start_time  # Calculate the elapsed time
    st.write(f"Model Training Time: {elapsed_time:.2f} seconds")  # Display the training time


def display_data(dn):
    st.title('Dataset Reviewer')
    df = load_data(f'data/{dn}.arff')
    selected_column = st.selectbox('Select a column to visualize', df.columns)

    if df[selected_column].dtype == 'object':
        # If the column is categorical, display a count plot
        fig, ax = plt.subplots()
        sns.countplot(y=selected_column, data=df, ax=ax)
        st.pyplot(fig)
    else:
        # If the column is numerical, display a histogram
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], kde=True, ax=ax)
        st.pyplot(fig)


def display_accuracy(dataset_name, model_name, test_accuracy, test_f1, train_accuracy, train_f1):
    st.write("## Results")
    st.write(f"**Dataset:** {dataset_name}")
    st.write(f"**Model:** {model_name}")

    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score"],
        "Train": [train_accuracy, train_f1],
        "Test": [test_accuracy, test_f1]
    })
    results_df = results_df.set_index("Metric")
    st.table(results_df)


def display_cm(y_test, y_pred, case):
    cm = confusion_matrix(y_test, y_pred)

    x = ['Predicted False', 'Predicted True']
    y = ['Actual False', 'Actual True']

    fig = go.Figure(data=go.Heatmap(z=cm,
                                    x=x,
                                    y=y,
                                    colorscale='Blues',
                                    showscale=True))

    # Add Text Annotations
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            fig.add_annotation(
                go.layout.Annotation(
                    x=x[j],
                    y=y[i],
                    xref='x',
                    yref='y',
                    text=str(value),
                    showarrow=False,
                    font=dict(size=16, color='black')
                )
            )

    fig.update_layout(
        title=f'{case} Confusion Matrix'
    )

    st.plotly_chart(fig)


def main():
    dataset_name, model_name = init_sidebar()

    x, y = load_and_preprocess(dataset_name)
    display_data(dataset_name)

    model = models[model_name]
    if x is not None:
        x_train, x_test, y_train, y_test = split_data(x, y)
        train(model, x_train, y_train, x_test, y_test)

        y_train_pred = model.predict(x_train, y_train)
        y_test_pred = model.predict(x_test, y_test)

        train_accuracy, train_f1 = model.evaluate_predictions(y_train_pred, y_train)
        test_accuracy, test_f1 = model.evaluate_predictions(y_test_pred, y_test)

        display_accuracy(dataset_name, model_name, test_accuracy, test_f1, train_accuracy, train_f1)

        display_cm(y_train, y_train_pred, "Train")
        display_cm(y_test, y_test_pred, "Test")
    else:
        st.error(f"The dataset {dataset_name} is not valid.")


if __name__ == "__main__":
    main()
