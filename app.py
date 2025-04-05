import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# DeepSeek API Key (store securely)
DEEPSEEK_API_KEY = "sk-a2c6864177364872be1fb0831d5b567f"

# Load the dataset
DATA_PATH = "data/Final_men_soccer.xlsx"  # Replace with your dataset path
df = pd.read_excel(DATA_PATH)

# Function to get AI response for a given question
def chat_with_data(user_message, data_context):
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers questions about the dataset and provides insights and visualizations."},
        {"role": "user", "content": f"{user_message}\n\nData:\n{data_context}"}
    ]
    payload = {"model": "deepseek-chat", "messages": messages}
    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Function to get AI-generated plot instructions
def get_ai_plot_instructions(data_context, user_question):
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    prompt = f"""
Analyze the dataset and, based on the following user question, determine the best visualizations that answer it.
User Question: "{user_question}"
Return ONLY valid JSON in the following exact structure:
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
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()

    try:
        ai_response = result["choices"][0]["message"]["content"]
        json_start = ai_response.find("{")
        json_end = ai_response.rfind("}") + 1
        ai_json = ai_response[json_start:json_end]
        return json.loads(ai_json)
    except (json.JSONDecodeError, KeyError, IndexError):
        st.error("⚠ AI returned invalid plot instructions. Debug output:")
        st.text(ai_response)
        return {"plots": []}

# Function to generate plots based on AI instructions
def generate_ai_plots(df, instructions):
    plots = instructions.get("plots", [])
    for plot in plots:
        p_type = plot.get("type", "").lower()
        cols = plot.get("columns", [])
        title = plot.get("title", "Untitled Plot")

        # Check if columns exist in dataframe
        if not cols or not all(c in df.columns for c in cols):
            st.warning(f"⚠ Invalid columns provided: {cols}")
            continue

        try:
            if p_type == "histogram" and len(cols) == 1:
                fig = px.histogram(df, x=cols[0], title=title, color=cols[0])
                st.plotly_chart(fig, use_container_width=True)

            elif p_type == "scatter" and len(cols) >= 2:
                fig = px.scatter(df, x=cols[0], y=cols[1], title=title, color=cols[1] if len(cols) > 1 else None)
                st.plotly_chart(fig, use_container_width=True)

            elif p_type == "box" and len(cols) == 1:
                fig = px.box(df, y=cols[0], title=title)
                st.plotly_chart(fig, use_container_width=True)

            elif p_type == "bar":
                if len(cols) == 1:
                    col = cols[0]
                    # Single-column bar chart using value counts
                    if df[col].dtype == object or df[col].nunique() < 30:
                        data = df[col].value_counts().nlargest(10).reset_index()
                        data.columns = [col, "count"]
                        fig = px.bar(data, x=col, y="count", title=title, color=col)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"⚠ Too many unique values in {col} for a bar chart.")
                elif len(cols) == 2:
                    x_col, y_col = cols
                    # Ensure x is categorical and y is numeric by converting if necessary
                    if x_col in df.columns and y_col in df.columns:
                        if df[x_col].dtype != object:
                            df[x_col] = df[x_col].astype(str)
                        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                        # Aggregate data by the categorical column (x_col)
                        group_data = df.groupby(x_col)[y_col].sum(numeric_only=True).nlargest(10).reset_index()
                        fig = px.bar(group_data, x=x_col, y=y_col, title=title, color=x_col)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"⚠ Columns {cols} not found in data.")
                else:
                    st.warning(f"⚠ Bar chart requires 1 or 2 columns. Got: {cols}")

            elif p_type == "heatmap":
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1:
                    correlation_matrix = df[numeric_cols].corr()
                    st.write(title)
                    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="coolwarm")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠ Not enough numeric columns for a heatmap.")

            else:
                st.warning(f"⚠ Unsupported or invalid plot type: {p_type}")

        except Exception as e:
            st.error(f"❌ Error generating {p_type} plot: {e}")

# Streamlit UI
st.set_page_config(page_title="AI Chat with Visuals", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📈 Chat with Your Data</h1>", unsafe_allow_html=True)

user_message = st.text_input("💬 Ask a question about your dataset:")

if st.button("Ask AI"):
    with st.spinner("Processing your query..."):
        sample_data = df.to_string(index=False)
        # Get AI text response (insights)
        ai_response = chat_with_data(user_message, sample_data)
        st.markdown("### 🤖 AI Response")
        st.write(ai_response)
        # Get and render visualizations
        st.markdown("### 📊 Visualizations")
        plot_instructions = get_ai_plot_instructions(sample_data, user_message)
        generate_ai_plots(df, plot_instructions)
