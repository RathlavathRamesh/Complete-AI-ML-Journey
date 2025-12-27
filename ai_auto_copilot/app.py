import streamlit as st
import pandas as pd
from agents import classify_intent, generate_reply
from whatsapp import send_whatsapp

st.title("📲 Coaching Sales AI Copilot")

msg = st.text_input("Incoming Student WhatsApp Message")

if st.button("Process Lead"):
    intent = classify_intent(msg)
    reply = generate_reply(msg)

    send_whatsapp(reply)

    try:
        df = pd.read_excel("leads.xlsx")
    except:
        df = pd.DataFrame(columns=["Message", "Intent"])

    df.loc[len(df)] = [msg, intent]
    df.to_excel("leads.xlsx", index=False)

    st.success("Lead processed & replied")
    st.write("Intent:", intent)
    st.write("Reply:", reply)
