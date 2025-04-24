import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import tiktoken
import stripe
import hashlib
import sqlite3
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables (optional since we're using direct API keys)
load_dotenv()

# ----------------------------
# Set Streamlit Page Configuration (Must be the first Streamlit command!)
# ----------------------------
st.set_page_config(page_title="AI Chat with Visuals", layout="wide")

# ----------------------------
# Configuration and API Keys
# ----------------------------

# API Keys (direct approach)
DEEPSEEK_API_KEY = "sk-65f90630f4954f0baa416fe1ec29fc83"
STRIPE_SECRET_KEY = "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"
STRIPE_PUBLISHABLE_KEY = "pk_live_51RFGlsBUCOYmcUSmxNXYazcqkJFhTWb8nn6Wgqb9AIxvB1D8Pd4bTHp0QHmEv73M7HZyTik9pg7fV8nqlNtpQyFI00fkrU2hds"

# Initialize Stripe
stripe.api_key = STRIPE_SECRET_KEY

# Define price IDs for your Stripe products
PRICE_IDS = {
    "Pricing 1": "price_1RHN7wBUCOYmcUSm1z2cyYtg",  # $10 Basic Plan 
    "Pricing 2": "price_1RHN96BUCOYmcUSmrPAui98G",  # $20 Standard Plan
    "Pricing 3": "price_1RHN8EBUCOYmcUSm9R8A2wph"   # $30 Premium Plan
}

# Load the dataset
DATA_PATH = "data/Sajjad_data.xlsx"  # Update with your dataset file path
df = pd.read_excel(DATA_PATH)

# Clean up duplicate columns and column names
df = df.loc[:, ~df.columns.duplicated()]
df.columns = df.columns.str.strip()

# ----------------------------
# Database Setup for User Management
# ----------------------------
def setup_db():
    """Set up SQLite database for user management."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        plan TEXT DEFAULT 'Free',
        plan_start_date TEXT,
        plan_end_date TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Call setup
setup_db()

# ----------------------------
# User Authentication Functions
# ----------------------------
def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user."""
    return stored_password == hash_password(provided_password)

def create_user(username, password, email):
    """Create a new user in the database."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                  (username, password_hash, email))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    
    return success

def authenticate_user(username, password):
    """Authenticate a user by username and password."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    c.execute("SELECT username, password_hash, plan FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and verify_password(user[1], password):
        return {"authenticated": True, "username": user[0], "plan": user[2]}
    else:
        return {"authenticated": False}

def update_user_plan(username, plan):
    """Update a user's subscription plan."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Set plan start date to today and end date to one month from today
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    c.execute("UPDATE users SET plan = ?, plan_start_date = ?, plan_end_date = ? WHERE username = ?",
              (plan, start_date, end_date, username))
    conn.commit()
    conn.close()

def get_user_plan(username):
    """Get a user's current plan."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    c.execute("SELECT plan FROM users WHERE username = ?", (username,))
    plan = c.fetchone()
    conn.close()
    
    return plan[0] if plan else "Free"

def get_user_email(username):
    """Get a user's email."""
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    c.execute("SELECT email FROM users WHERE username = ?", (username,))
    email = c.fetchone()
    conn.close()
    
    return email[0] if email else ""

# ----------------------------
# Stripe Payment Functions
# ----------------------------
def create_checkout_session(price_id, user_email, plan_name):
    """Create a Stripe checkout session."""
    try:
        # Create checkout session with simple parameters
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:8501/?success=true&plan=' + plan_name,
            cancel_url='http://localhost:8501/?cancel=true',
            customer_email=user_email,
            metadata={
                'username': st.session_state.username,
                'plan': plan_name
            }
        )
        return checkout_session
    except Exception as e:
        st.error(f"Error creating checkout session: {str(e)}")
        return None

def create_all_checkout_links(username):
    """Create all checkout links upfront for a user."""
    links = {}
    email = get_user_email(username)
    
    if not email:
        st.error("Unable to retrieve email for user. Please update your profile.")
        return {}
    
    for plan_key, plan_name in [("Pricing 1", "Basic"), ("Pricing 2", "Standard"), ("Pricing 3", "Premium")]:
        try:
            session = create_checkout_session(PRICE_IDS[plan_key], email, plan_key)
            if session:
                links[plan_key] = session.url
        except Exception as e:
            st.error(f"Error creating {plan_name} checkout: {str(e)}")
            links[plan_key] = "#"
    
    return links

# ----------------------------
# Initialize Session State
# ----------------------------
def init_session_state():
    """Initialize session state variables."""
    if 'tokens_used' not in st.session_state:
        st.session_state.tokens_used = 0
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = ""
    
    if 'pricing_plan' not in st.session_state:
        st.session_state.pricing_plan = "Free"
    
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = "login"  # Can be 'login' or 'register'

# Call initialization
init_session_state()

# ----------------------------
# Authentication UI
# ----------------------------
def show_login_page():
    """Display the login page."""
    st.header("Login")
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Login"):
            result = authenticate_user(username, password)
            if result["authenticated"]:
                st.session_state.logged_in = True
                st.session_state.username = result["username"]
                st.session_state.pricing_plan = result["plan"]
                st.success("Successfully logged in!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    
    with col2:
        if st.button("Create Account"):
            st.session_state.auth_page = "register"
            st.rerun()

def show_register_page():
    """Display the registration page."""
    st.header("Create Account")
    
    username = st.text_input("Username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success = create_user(username, password, email)
                if success:
                    st.success("Account created successfully! Please log in.")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error("Username or email already exists.")
    
    with col2:
        if st.button("Back to Login"):
            st.session_state.auth_page = "login"
            st.rerun()

# ----------------------------
# Plans and Payment UI
# ----------------------------
def show_plans_page():
    """Display the subscription plans page."""
    st.header("Choose a Subscription Plan")
    
    # Test Stripe connection
    try:
        stripe.Balance.retrieve()
        st.success("✅ Connected to Stripe API successfully!")
    except Exception as e:
        st.error(f"❌ Error connecting to Stripe: {str(e)}")
    
    # Create all checkout links upfront
    checkout_links = create_all_checkout_links(st.session_state.username)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Free")
        st.write("Basic access")
        st.write("1 filter (sport only)")
        st.write("Price: $0/month")
        if st.button("Current Plan" if st.session_state.pricing_plan == "Free" else "Select Free"):
            st.session_state.pricing_plan = "Free"
            update_user_plan(st.session_state.username, "Free")
            st.success("You are now on the Free plan!")
            st.rerun()
    
    with col2:
        st.subheader("Basic")
        st.write("2 filters including sport")
        st.write("Price: $10/month")
        if st.session_state.pricing_plan == "Pricing 1":
            st.success("Current Plan")
        elif "Pricing 1" in checkout_links:
            st.markdown(f'''
            <a href="{checkout_links['Pricing 1']}" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">
                Upgrade to Basic
            </a>
            ''', unsafe_allow_html=True)
    
    with col3:
        st.subheader("Standard")
        st.write("3 filters including academic")
        st.write("Price: $20/month")
        if st.session_state.pricing_plan == "Pricing 2":
            st.success("Current Plan")
        elif "Pricing 2" in checkout_links:
            st.markdown(f'''
            <a href="{checkout_links['Pricing 2']}" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">
                Upgrade to Standard
            </a>
            ''', unsafe_allow_html=True)
    
    with col4:
        st.subheader("Premium")
        st.write("All filters")
        st.write("Price: $30/month")
        if st.session_state.pricing_plan == "Pricing 3":
            st.success("Current Plan")
        elif "Pricing 3" in checkout_links:
            st.markdown(f'''
            <a href="{checkout_links['Pricing 3']}" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">
                Upgrade to Premium
            </a>
            ''', unsafe_allow_html=True)

# ----------------------------
# Success & Cancel Pages
# ----------------------------
def handle_success():
    """Handle the success page after payment."""
    if "plan" in st.query_params:
        plan = st.query_params["plan"]
        update_user_plan(st.session_state.username, plan)
        st.session_state.pricing_plan = plan
        st.success(f"Payment successful! You are now on the {plan} plan.")
        if st.button("Continue to Dashboard"):
            st.query_params.clear()
            st.rerun()

def handle_cancel():
    """Handle the cancel page after payment cancellation."""
    st.error("Payment was cancelled.")
    if st.button("Return to Dashboard"):
        st.query_params.clear()
        st.rerun()

# ----------------------------
# Handle URL Parameters
# ----------------------------
def handle_url_params():
    """Handle URL parameters for payment success/cancel."""
    if "success" in st.query_params:
        handle_success()
        return True
    elif "cancel" in st.query_params:
        handle_cancel()
        return True
    return False

# ----------------------------
# User Dashboard
# ----------------------------
def show_user_dashboard():
    """Display the user dashboard."""
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    st.sidebar.write(f"Current Plan: {st.session_state.pricing_plan}")
    
    if st.sidebar.button("Manage Subscription"):
        show_plans_page()
        return True
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.pricing_plan = "Free"
        st.rerun()
    
    return False

# ----------------------------
# Sidebar: Pricing Plan and Filters
# ----------------------------
def show_filters():
    """Display and handle the filter sidebar."""
    st.sidebar.header("Filters")
    
    pricing_plan = st.session_state.pricing_plan
    
    # Token optimization: cost and monthly cap per plan
    pricing_cost = {"Free": 10, "Pricing 1": 20, "Pricing 2": 30, "Pricing 3": 40}
    monthly_cap = {"Free": 50, "Pricing 1": 200, "Pricing 2": 300, "Pricing 3": 500}
    
    # Common filters helper
    def multiselect_filter(label: str, column: str):
        return st.sidebar.multiselect(label, options=df[column].unique())
    
    # Apply filters based on pricing plan
    filtered_df = df.copy()
    
    # Free plan (only sport filter)
    if pricing_plan == "Free":
        sport_filter = multiselect_filter("Sport", "Sport")
        if sport_filter:
            filtered_df = filtered_df[filtered_df["Sport"].isin(sport_filter)]
    
    # Price 1 ($10): 2 filters including sport
    elif pricing_plan == "Pricing 1":
        sport_filter = multiselect_filter("Sport", "Sport")
        gender_filter = multiselect_filter("Gender", "Gender")
        
        if sport_filter:
            filtered_df = filtered_df[filtered_df["Sport"].isin(sport_filter)]
        if gender_filter:
            filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]
    
    # Price 2 ($20): 3 filters including academic
    elif pricing_plan == "Pricing 2":
        sport_filter = multiselect_filter("Sport", "Sport")
        gender_filter = multiselect_filter("Gender", "Gender")
        academics_filter = multiselect_filter("Academics", "Academics")
        
        if sport_filter:
            filtered_df = filtered_df[filtered_df["Sport"].isin(sport_filter)]
        if gender_filter:
            filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]
        if academics_filter:
            filtered_df = filtered_df[filtered_df["Academics"].isin(academics_filter)]
    
    # Price 3 ($30): All filters
    else:  # Pricing 3
        region_filter = multiselect_filter("Region", "Region")
        division_filter = multiselect_filter("Division", "Division")
        sat_range_filter = multiselect_filter("SAT Range", "SAT range")
        gender_filter = multiselect_filter("Gender", "Gender")
        position_filter = multiselect_filter("Position", "Position")
        academic_year_filter = multiselect_filter("Academic Year", "Academic Year")
        sport_filter = multiselect_filter("Sport", "Sport")
        academics_filter = multiselect_filter("Academics", "Academics")
        
        players = sorted([n for n in df["Player Name"].unique() if isinstance(n, str)])
        player_filter = st.sidebar.multiselect("Select Player(s)", options=players)
        
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
        if sport_filter:
            filtered_df = filtered_df[filtered_df["Sport"].isin(sport_filter)]
        if academics_filter:
            filtered_df = filtered_df[filtered_df["Academics"].isin(academics_filter)]
        if player_filter:
            filtered_df = filtered_df[filtered_df["Player Name"].isin(player_filter)]
    
    return filtered_df, pricing_cost[pricing_plan], monthly_cap[pricing_plan]

# ----------------------------
# Helper: Count tokens for DeepSeek model
# ----------------------------
def count_tokens(text: str, model: str = "deepseek-chat") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ----------------------------
# Functions for AI Responses & Visualizations
# ----------------------------
def chat_with_data(user_message: str, data_context: str) -> str | None:
    total_tokens = count_tokens(user_message + data_context)
    if total_tokens > 60000:
        st.error(f"⚠ Data context too large ({total_tokens} tokens). Please apply more filters to reduce the data size before asking the AI.")
        return None

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers questions about the dataset and provides insights and visualizations."},
        {"role": "user",   "content": f"{user_message}\n\nData:\n{data_context}"}
    ]
    payload = {"model": "deepseek-chat", "messages": messages}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        err = None
        try:
            err = response.json().get("error", {}).get("message", "Unknown error")
        except json.JSONDecodeError:
            err = response.text
        if err and "exceed" in err.lower():
            st.error("⚠ Token limit exceeded. Please apply more filters to reduce the data size and try again.")
        else:
            st.error(f"⚠ API Error: {err}")
        return None

    try:
        result = response.json()
        st.session_state.tokens_used += pricing_cost[st.session_state.pricing_plan]
        return result["choices"][0]["message"]["content"]
    except Exception:
        st.error("⚠ Error processing AI response.")
        return None


def get_ai_plot_instructions(data_context: str, user_question: str) -> dict:
    total_tokens = count_tokens(user_question + data_context)
    if total_tokens > 60000:
        st.error(f"⚠ Data context too large ({total_tokens} tokens). Please apply more filters before generating plots.")
        return {"plots": []}

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    prompt = f"""
Analyze the dataset and, based on the following user question, determine the best visualizations that answer it.
User Question: \"{user_question}\" 
Return ONLY valid JSON in the exact structure:
{{
    "plots": [
        {{"type": "histogram", "columns": ["Column1"], "title": "Title"}},
        {{"type": "scatter",   "columns": ["ColumnX","ColumnY"], "title": "Scatter Title"}}
    ]
}}
Do not include explanations or extra text. Just return the JSON.
Here is a sample of the dataset:
{data_context}
"""
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        err = None
        try:
            err = response.json().get("error", {}).get("message", "Unknown error")
        except json.JSONDecodeError:
            err = response.text
        if err and "exceed" in err.lower():
            st.error("⚠ Token limit exceeded. Please apply more filters to reduce the data size and try again.")
        else:
            st.error(f"⚠ API Error: {err}")
        return {"plots": []}

    try:
        text = response.json()["choices"][0]["message"]["content"]
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        return json.loads(text[json_start:json_end])
    except Exception:
        st.error("⚠ AI returned invalid plot instructions.")
        return {"plots": []}


def generate_ai_plots(dataframe: pd.DataFrame, instructions: dict) -> None:
    for plot in instructions.get("plots", []):
        p_type = plot.get("type", "").lower()
        cols = plot.get("columns", [])
        title = plot.get("title", "Untitled Plot")
        if not cols or any(c not in dataframe.columns for c in cols):
            st.warning(f"⚠ Invalid columns: {cols}")
            continue
        try:
            if p_type == "histogram" and len(cols) == 1:
                fig = px.histogram(dataframe, x=cols[0], title=title, color=cols[0])
            elif p_type == "scatter" and len(cols) >= 2:
                fig = px.scatter(dataframe, x=cols[0], y=cols[1], title=title, color=cols[1])
            elif p_type == "box" and len(cols) == 1:
                fig = px.box(dataframe, y=cols[0], title=title)
            elif p_type == "bar":
                if len(cols) == 1:
                    cnt = dataframe[cols[0]].value_counts().nlargest(10).reset_index()
                    cnt.columns = [cols[0], "count"]
                    fig = px.bar(cnt, x=cols[0], y="count", title=title, color=cols[0])
                else:
                    x, y = cols[0], cols[1]
                    grp = dataframe.groupby(x)[y].sum(numeric_only=True).nlargest(10).reset_index()
                    fig = px.bar(grp, x=x, y=y, title=title, color=x)
            else:
                st.warning(f"⚠ Unsupported plot type: {p_type}")
                continue
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠ Error generating plot: {e}")

# ----------------------------
# Main Application Flow
# ----------------------------
def main():
    """Main application flow."""
    # Check for URL parameters (success/cancel from Stripe)
    if handle_url_params():
        return
    
    # Authentication flow
    if not st.session_state.logged_in:
        if st.session_state.auth_page == "login":
            show_login_page()
        else:
            show_register_page()
        return
    
    # User dashboard (returns True if we're showing the plans page)
    if show_user_dashboard():
        return
    
    # Get filtered data based on user's plan
    filtered_df, current_cost, monthly_cap_val = show_filters()
    
    # Token/Spending Control
    if st.session_state.tokens_used + current_cost > monthly_cap_val:
        st.error("Monthly spending cap exceeded. Please wait until the next billing cycle.")
        return
    
    # Main Interaction
    st.title("AI-Powered Dataset Insights")
    
    user_question = st.text_input("Ask a question about the data:")
    if user_question:
        st.write(f"Question: {user_question}")
        data_context = filtered_df.to_string(index=False)
        
        # Chat response
        ai_response = chat_with_data(user_question, data_context)
        if ai_response:
            st.write(f"AI Response: {ai_response}")
            
            # Visualization instructions + rendering
            ai_plots = get_ai_plot_instructions(data_context, user_question)
            if ai_plots.get("plots"):
                generate_ai_plots(filtered_df, ai_plots)
            else:
                st.write("No plots available for this question.")

# Run the application
if __name__ == "__main__":
    main()
