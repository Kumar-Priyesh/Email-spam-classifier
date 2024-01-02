import streamlit as st
from predict import predict_message


def main():
    st.title("Email Spam classifier")
    input_email = st.text_area('Enter your Email')

    button_check = st.button('Predict')

    if button_check:
        if not input_email:
            st.warning("Please type the email content.")
        else:
            prediction = predict_message(input_email)

            if prediction == 0:
                st.success('Not spam')
            else:
                st.success('Spam')



if __name__ == "__main__":
    main()