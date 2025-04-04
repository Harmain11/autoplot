import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# DeepSeek API Key (store securely)
DEEPSEEK_API_KEY = "sk-a2c6864177364872be1fb0831d5b567f"

# Function to get AI insights
def get_ai_insights(data_context):
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", 
               "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    
    prompt = f"""Analyze the dataset and provide key insights.
Data:
{data_context}"""
    
    payload = {"model": "deepseek-chat", 
               "messages": [{"role": "user", "content": prompt}]}
    
    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()
    
    return result["choices"][0]["message"]["content"]

# Function to get AI plot instructions in JSON format
def get_ai_plot_instructions(data_context):
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", 
               "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    
    prompt = f"""
Analyze the dataset and determine the best visualizations. 
Return **only** valid JSON in the following exact structure:
{{
    "plots": [
        {{"type": "histogram", "columns": ["Column1"], "title": "Title"}},
        {{"type": "scatter", "columns": ["ColumnX", "ColumnY"], "title": "Scatter Title"}}
    ]
}}
Do not include explanations or extra text. Just return the JSON.
Here is a sample of the dataset:
{data_context}
"""
    
    payload = {"model": "deepseek-chat", 
               "messages": [{"role": "user", "content": prompt}]}
    
    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()

    # Extract response safely
    try:
        ai_response = result["choices"][0]["message"]["content"]
        json_start = ai_response.find("{")  
        json_end = ai_response.rfind("}") + 1
        ai_json = ai_response[json_start:json_end]  
        plot_instructions = json.loads(ai_json)  # Convert string to dict
        return plot_instructions
    except (json.JSONDecodeError, KeyError, IndexError):
        st.error("⚠ AI returned invalid JSON. Debugging output:")
        st.text(ai_response)
        return {"plots": []}  # Prevent crash

# Function to generate plots based on AI instructions
def generate_ai_plots(df, instructions):
    st.write("### 📊 AI-Generated Visualizations")
    plots = instructions.get("plots", [])
    
    for plot in plots:
        p_type = plot.get("type", "").lower()
        cols = plot.get("columns", [])
        title = plot.get("title", "Untitled Plot")
        
        if p_type == "histogram" and len(cols) == 1 and cols[0] in df.columns:
            fig = px.histogram(df, x=cols[0], title=title, color=cols[0])
            st.plotly_chart(fig)
        
        elif p_type == "scatter" and len(cols) == 2 and all(c in df.columns for c in cols):
            fig = px.scatter(df, x=cols[0], y=cols[1], title=title, color=cols[1])
            st.plotly_chart(fig)
        
        elif p_type == "box" and len(cols) == 1 and cols[0] in df.columns:
            fig = px.box(df, y=cols[0], title=title, color=cols[0])
            st.plotly_chart(fig)
        
        elif p_type == "bar" and len(cols) == 1 and cols[0] in df.columns:
            if df[cols[0]].dtype == object or df[cols[0]].nunique() < 20:
                data = df[cols[0]].value_counts().reset_index()
                data.columns = [cols[0], "count"]
                fig = px.bar(data, x=cols[0], y="count", title=title, color=cols[0])
                st.plotly_chart(fig)
            else:
                st.write(f"Skipping bar chart for {cols[0]} as it may not be categorical.")
        
        elif p_type == "heatmap":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                st.write(title)
                fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="coolwarm")
                st.plotly_chart(fig)
            else:
                st.write("Not enough numeric columns for a heatmap.")
        else:
            st.write(f"Could not generate plot for instruction: {plot}")

# Function to chat with data
def chat_with_data(user_message, data_context):
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", 
               "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers questions about the dataset."},
        {"role": "user", "content": f"{user_message}\nData:\n{data_context}"}
    ]
    
    payload = {"model": "deepseek-chat", "messages": messages}
    
    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()
    
    return result["choices"][0]["message"]["content"]

# Streamlit UI with Tabs and Improved UI
st.set_page_config(page_title="AI-Powered Auto Visualization", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI-Powered Auto Visualization & Chat</h1>", unsafe_allow_html=True)

# Sidebar for file upload and basic instructions
st.sidebar.markdown("## Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file, engine="openpyxl")

    st.markdown("### 🔍 Data Preview")
    st.dataframe(df)

    # Create Tabs for AI Insights and Chat with Data
    tab1, tab2 = st.tabs(["🤖 AI Insights", "💬 Chat with Data"])

    with tab1:
        st.subheader("🧠 AI-Generated Insights")
        if st.button("Generate AI Insights", key="insights"):
            with st.spinner("AI is analyzing..."):
                data_context = df.head(10).to_string()
                insights = get_ai_insights(data_context)
                st.markdown(f"**Insights:** {insights}")

    with tab2:
        st.subheader("💬 Chat with Your Data")
        user_message = st.text_input("Ask a question about your dataset:")
        if st.button("Ask AI", key="chat"):
            with st.spinner("AI is thinking..."):
                data_context = df.head(10).to_string()
                bot_response = chat_with_data(user_message, data_context)
                st.markdown("#### 🤖 AI Response:")
                st.write(bot_response)
