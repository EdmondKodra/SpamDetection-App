import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# ========== Data ==============
data = pd.read_csv("Spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# ========== Model ==============
model = MultinomialNB()
model.fit(features, cat_train)

# ========== Prediction Function ==============
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# ========== Streamlit UI ==============
st.set_page_config(page_title="Spam Detection App", page_icon="📧", layout="centered")

# Custom CSS 
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .title {
        text-align: center;
        font-size: 35px !important;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>📧 Spam Detection App</h1>", unsafe_allow_html=True)
st.write("Shkruaj një mesazh më poshtë dhe modeli do ta klasifikojë si **Spam** ose **Not Spam**.")

# Input
input_mess = st.text_area("✉️ Enter Message Here:", height=120)

if st.button("🔍 Validate"):
    if input_mess.strip() == "":
        st.warning("Ju lutem shkruani një mesazh!")
    else:
        output = predict(input_mess)

        if output == "Spam":
            st.error("🚨 Ky mesazh është **Spam**!")
        else:
            st.success("✅ Ky mesazh është **Not Spam** (mesazh normal).")
