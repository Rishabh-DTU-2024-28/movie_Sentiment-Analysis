import streamlit as st
import joblib


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# -----------------------------------------
# LOAD MODEL AND VECTORIZER
# -----------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("best_movie_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("🎬 Movie Sentiment Analysis")

st.write(
    "Enter a movie review below and the model will predict "
    "whether the sentiment is Positive or Negative."
)

st.divider()


# -----------------------------------------
# USER INPUT
# -----------------------------------------

review = st.text_area(
    "✍️ Enter your movie review:",
    placeholder="Example: This movie was amazing. I really enjoyed it!",
    height=150
)


# -----------------------------------------
# PREDICTION BUTTON
# -----------------------------------------

if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        # Convert text into TF-IDF features
        review_vector = vectorizer.transform([review])

        # Make prediction
        prediction = model.predict(review_vector)[0]

        # ---------------------------------
        # DISPLAY RESULT
        # ---------------------------------

        st.subheader("Prediction")

        if prediction == 1 or str(prediction).lower() == "positive":

            st.success("😊 Positive Review")

        else:

            st.error("😞 Negative Review")


# -----------------------------------------
# EXAMPLE REVIEWS
# -----------------------------------------

st.divider()

st.subheader("💡 Example Reviews")

st.write("**Positive:**")
st.info(
    "This movie was absolutely fantastic. "
    "The acting and story were amazing."
)

st.write("**Negative:**")
st.info(
    "The movie was boring and disappointing. "
    "I did not enjoy it at all."
)


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption("Movie Sentiment Analysis | Machine Learning Project")