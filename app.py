
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.common import DATA, MODEL, ROOT, load_data
from src.predict import predict_one
from src.train import main as train_model


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    layout="wide",
    page_icon="⚙️"
)


# AUTOMATIC MODEL TRAINING
@st.cache_resource(
    show_spinner="Preparing machine learning model..."
)
def prepare_model():
    if DATA.exists() and not MODEL.exists():
        train_model()
    return True


prepare_model()


# MAIN DASHBOARD
st.title("⚙️ Enterprise Predictive Maintenance Intelligence")

st.caption(
    "Portfolio demo using synthetic UCI AI4I 2020 data — "
    "not validated for industrial safety decisions."
)

tab1, tab2, tab3 = st.tabs([
    "Overview & EDA",
    "Failure Prediction",
    "Model Performance"
])


# =====================================
# TAB 1 - OVERVIEW AND EDA
# =====================================

with tab1:

    if not DATA.exists():
        st.info(
            "Dataset missing. Run python download_data.py"
        )

    else:
        df = load_data()

        a, b, c = st.columns(3)

        a.metric(
            "Observations",
            f"{len(df):,}"
        )

        b.metric(
            "Failure cases",
            int(df["Machine failure"].sum())
        )

        c.metric(
            "Failure rate",
            f"{df['Machine failure'].mean():.2%}"
        )

        st.subheader("Failure distribution")

        st.bar_chart(
            df["Machine failure"]
            .map({
                0: "No failure",
                1: "Failure"
            })
            .value_counts()
        )

        st.subheader("Sensor measurements")

        col = st.selectbox(
            "Select sensor",
            [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        )

        fig, ax = plt.subplots()

        ax.hist(
            df.loc[df["Machine failure"] == 0, col],
            bins=30,
            alpha=0.65,
            label="No failure"
        )

        ax.hist(
            df.loc[df["Machine failure"] == 1, col],
            bins=30,
            alpha=0.65,
            label="Failure"
        )

        ax.legend()
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            df.head(20),
            width="stretch"
        )


# =====================================
# TAB 2 - FAILURE PREDICTION
# =====================================

with tab2:

    if not MODEL.exists():

        st.warning(
            "Model not available. "
            "Check dataset and model training."
        )

    else:

        with st.form("predict"):

            a, b = st.columns(2)

            with a:

                typ = st.selectbox(
                    "Product type",
                    ["L", "M", "H"]
                )

                air = st.number_input(
                    "Air temperature (K)",
                    value=300.0
                )

                proc = st.number_input(
                    "Process temperature (K)",
                    value=310.0
                )

            with b:

                rpm = st.number_input(
                    "Rotational speed (rpm)",
                    value=1500,
                    min_value=0
                )

                torque = st.number_input(
                    "Torque (Nm)",
                    value=40.0,
                    min_value=0.0
                )

                wear = st.number_input(
                    "Tool wear (min)",
                    value=100,
                    min_value=0
                )

            submitted = st.form_submit_button(
                "Predict failure risk"
            )

        if submitted:

            result = predict_one({
                "Type": typ,
                "Air temperature [K]": air,
                "Process temperature [K]": proc,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear
            })

            probability = result["failure_probability"]

            st.metric(
                "Estimated failure probability",
                f"{probability:.6%}"
            )

            # DEMONSTRATION RISK LEVELS

            if probability < 0.30:

                st.success(
                    "🟢 LOW RISK - Low estimated failure risk"
                )

            elif probability < 0.70:

                st.warning(
                    "🟠 MEDIUM RISK - Inspection review suggested"
                )

            else:

                st.error(
                    "🔴 HIGH RISK - Maintenance review suggested"
                )

            st.write(
                "Classification:",
                "Potential failure"
                if result["predicted_failure"]
                else "No failure predicted"
            )

            st.caption(
                f"Decision threshold: "
                f"{result['decision_threshold']:.3f}. "
                "Research demo; not a certified failure forecast. "
                "Risk bands are illustrative."
            )


# =====================================
# TAB 3 - MODEL PERFORMANCE
# =====================================

with tab3:

    path = ROOT / "reports" / "metrics.json"

    if not path.exists():

        st.info(
            "Train the model to generate metrics."
        )

    else:

        m = json.loads(path.read_text())

        st.write(
            "Selected model:",
            m["selected_model"]
        )

        a, b, c = st.columns(3)

        a.metric(
            "Test F1",
            f"{m['test_f1']:.3f}"
        )

        b.metric(
            "Test recall",
            f"{m['test_recall']:.3f}"
        )

        c.metric(
            "Test PR-AUC (AP)",
            f"{m['test_average_precision']:.3f}"
        )

        st.dataframe(
            pd.DataFrame(
                m["validation_leaderboard"]
            ),
            width="stretch"
        )

        st.write(
            "Test confusion matrix [TN, FP; FN, TP]:",
            m["test_confusion_matrix"]
        )
