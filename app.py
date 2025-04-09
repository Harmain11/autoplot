import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# ----------------------------
# Set Streamlit Page Configuration (Must be the first Streamlit command!)
# ----------------------------
st.set_page_config(page_title="AI Chat with Visuals", layout="wide")

# ----------------------------
# Configuration and Data Load
# ----------------------------

# DeepSeek API Key (store securely)
DEEPSEEK_API_KEY = "sk-65f90630f4954f0baa416fe1ec29fc83"

# Load the new dataset (data remains on backend)
DATA_PATH = "data/Sajjad_data.xlsx"  # Update with your dataset file path
df = pd.read_excel(DATA_PATH)

# Remove duplicate columns (if any) and strip extra spaces from column names.
df = df.loc[:, ~df.columns.duplicated()]
df.columns = df.columns.str.strip()

# ----------------------------
# Sidebar: Pricing Plan and Filters
# ----------------------------
st.sidebar.header("User Settings & Filters")

# Pricing plan selection
pricing_plan = st.sidebar.selectbox("Select Pricing Plan", ["Free", "Pricing 1", "Pricing 2", "Pricing 3"])

# Token optimization: set up query cost and monthly spending cap per plan (adjust as needed)
pricing_cost = {"Free": 10, "Pricing 1": 20, "Pricing 2": 30, "Pricing 3": 40}
monthly_cap = {"Free": 50, "Pricing 1": 200, "Pricing 2": 300, "Pricing 3": 500}

# Sidebar: Common Filters (based on dataset columns)
region_filter = st.sidebar.multiselect("Region", options=df["Region"].unique())
division_filter = st.sidebar.multiselect("Division", options=df["Division"].unique())
sat_range_filter = st.sidebar.multiselect("SAT Range", options=df["SAT range"].unique())
gender_filter = st.sidebar.multiselect("Gender", options=df["Gender"].unique())
position_filter = st.sidebar.multiselect("Position", options=df["Position"].unique())
academic_year_filter = st.sidebar.multiselect("Academic Year", options=df["Academic Year"].unique())

# Pricing-dependent Filters with Player Name Option
if pricing_plan in ["Free", "Pricing 1"]:
    sport_filter = st.sidebar.multiselect("Sport", options=df["Sport"].unique())
    players = sorted([name for name in df["Player Name"].unique() if isinstance(name, str)])
    player_options = ["All"] + players
    player_filter = st.sidebar.selectbox("Select Player", options=player_options, index=0)
elif pricing_plan == "Pricing 2":
    sport_filter = st.sidebar.multiselect("Sport", options=df["Sport"].unique())
    academics_filter = st.sidebar.multiselect("Academics", options=df["Academics"].unique())
    players = sorted([name for name in df["Player Name"].unique() if isinstance(name, str)])
    player_options = ["All"] + players
    player_filter = st.sidebar.selectbox("Select Player", options=player_options, index=0)
elif pricing_plan == "Pricing 3":
    sport_filter = st.sidebar.multiselect("Sport", options=df["Sport"].unique())
    academics_filter = st.sidebar.multiselect("Academics", options=df["Academics"].unique())
    players = sorted([name for name in df["Player Name"].unique() if isinstance(name, str)])
    player_filter = st.sidebar.multiselect("Select Player(s)", options=players, default=players)

# ----------------------------
# Data Filtering
# ----------------------------
filtered_df = df.copy()

# Apply common filters
if region_filter:
    filtered_df = filtered_df[filtered_df["Region"].isin(region_filter)]
if division_filter:
    filtered_df = filtered_df[filtered_df["Division"].isin(division_filter)]
if sat_range_filter:
    filtered_df = filtered_df[filtered_df["SAT range"].isin(sat_range_filter)]
if gender_filter:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]
if position_filter:
    filtered_df = filtered_df[filtered_df["Position"].isin(position_filter)]
if academic_year_filter:
    filtered_df = filtered_df[filtered_df["Academic Year"].isin(academic_year_filter)]

# Apply pricing-specific filters
if pricing_plan in ["Free", "Pricing 1", "Pricing 2"]:
    if sport_filter:
        filtered_df = filtered_df[filtered_df["Sport"].isin(sport_filter)]
    if player_filter != "All":
        filtered_df = filtered_df[filtered_df["Player Name"] == player_filter]
elif pricing_plan == "Pricing 3":
    if sport_filter:
        filtered_df = filtered_df[filtered_df["Sport"].isin(sport_filter)]
    if academics_filter:
        filtered_df = filtered_df[filtered_df["Academics"].isin(academics_filter)]
    all_players = set([name for name in df["Player Name"].unique() if isinstance(name, str)])
    selected_players = set(player_filter)
    if selected_players != all_players:
        filtered_df = filtered_df[filtered_df["Player Name"].isin(player_filter)]

# ----------------------------
# Token/Spending Control
# ----------------------------
if 'tokens_used' not in st.session_state:
    st.session_state.tokens_used = 0

current_cost = pricing_cost[pricing_plan]
if st.session_state.tokens_used + current_cost > monthly_cap[pricing_plan]:
    st.error("Monthly spending cap exceeded. Please wait until the next billing cycle.")
    st.stop()

# ----------------------------
# Functions for AI Responses & Visualizations
# ----------------------------

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
    try:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No response content found.")
    except (json.JSONDecodeError, KeyError, IndexError):
        st.error("⚠ Error processing AI response. Debug output:")
        st.text(response.text)
        return "Error occurred while processing the response."

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
    try:
        result = response.json()
        ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No valid plot instructions.")
        json_start = ai_response.find("{")
        json_end = ai_response.rfind("}") + 1
        ai_json = ai_response[json_start:json_end]
        return json.loads(ai_json)
    except (json.JSONDecodeError, KeyError, IndexError):
        st.error("⚠ AI returned invalid plot instructions. Debug output:")
        st.text(response.text)
        return {"plots": []}

def generate_ai_plots(dataframe, instructions):
    plots = instructions.get("plots", [])
    for plot in plots:
        p_type = plot.get("type", "").lower()
        cols = plot.get("columns", [])
        title = plot.get("title", "Untitled Plot")
        if not cols or not all(c in dataframe.columns for c in cols):
            st.warning(f"⚠ Invalid columns provided: {cols}")
            continue
        try:
            if p_type == "histogram" and len(cols) == 1:
                fig = px.histogram(dataframe, x=cols[0], title=title, color=cols[0])
                st.plotly_chart(fig, use_container_width=True)
            elif p_type == "scatter" and len(cols) >= 2:
                fig = px.scatter(dataframe, x=cols[0], y=cols[1], title=title, color=cols[1] if len(cols) > 1 else None)
                st.plotly_chart(fig, use_container_width=True)
            elif p_type == "box" and len(cols) == 1:
                fig = px.box(dataframe, y=cols[0], title=title)
                st.plotly_chart(fig, use_container_width=True)
            elif p_type == "bar":
                if len(cols) == 1:
                    col = cols[0]
                    if dataframe[col].dtype == object or dataframe[col].nunique() < 30:
                        data = dataframe[col].value_counts().nlargest(10).reset_index()
                        data.columns = [col, "count"]
                        fig = px.bar(data, x=col, y="count", title=title, color=col)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"⚠ Too many unique values in {col} for a bar chart.")
                elif len(cols) == 2:
                    x_col, y_col = cols
                    if x_col in dataframe.columns and y_col in dataframe.columns:
                        if dataframe[x_col].dtype != object:
                            dataframe[x_col] = dataframe[x_col].astype(str)
                        dataframe[y_col] = pd.to_numeric(dataframe[y_col], errors='coerce')
                        group_data = dataframe.groupby(x_col)[y_col].sum(numeric_only=True).nlargest(10).reset_index()
                        fig = px.bar(group_data, x=x_col, y=y_col, title=title, color=x_col)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠ Unsupported plot type {p_type}.")
        except Exception as e:
            st.error(f"⚠ Error generating plot: {e}")
            continue

# ----------------------------
# Main Interaction
# ----------------------------

st.title("AI-Powered Dataset Insights")

user_question = st.text_input("Ask a question about the data:")
if user_question:
    st.write(f"Question: {user_question}")
    data_context = df.head(10).to_json(orient="split")  # Example of data context, change as necessary
    ai_response = chat_with_data(user_question, data_context)
    st.write(f"AI Response: {ai_response}")
    
    # Get AI plot instructions and generate plots
    ai_plots = get_ai_plot_instructions(data_context, user_question)
    if ai_plots.get("plots"):
        generate_ai_plots(filtered_df, ai_plots)
    else:
        st.write("No plots available for this question.")
