# -*- coding: utf-8 -*-
"""
data_analysis.ipynb

This script creates a Streamlit-based "Data Analysis Agent" that allows users
to upload a CSV and ask questions about it. The agent uses the OpenAI GPT-4o
model to understand queries, generate Python code (pandas/matplotlib),
execute it, and provide natural language explanations.
"""

import os, io, re
import pandas as pd
from openai import OpenAI
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Any


try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# ------------------  Agent Tools & Agents ---------------------------

def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on intent."""
    system_prompt = """
    You are an assistant that determines if a query is requesting a data visualization.
    Respond with only 'true' if the query is asking for a plot, chart, graph, or
    any visual representation of data. Otherwise, respond with 'false'.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    try:
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0.0,
            max_tokens=5
        )
        intent_response = response.choices[0].message.content.strip().lower()
        return intent_response == "true"
    except Exception as e:
        st.error(f"Error in QueryUnderstandingTool: {e}")
        return False

def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code."""
    return f"""
    Given a pandas DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas and matplotlib.pyplot (as plt) to answer:
    "{query}"

    Rules:
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (the matplotlib Figure object) to a variable named `result`.
    3. Create only ONE plot. Set `figsize=(6,4)`. Add a descriptive title and labels.
    4. Ensure the plot is self-contained and ready to be displayed. Add `plt.tight_layout()`.
    5. Return your answer inside a single markdown fence: ```python ... ```
    """

def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code."""
    return f"""
    Given a pandas DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using only pandas to answer:
    "{query}"

    Rules:
    1. Use pandas operations on the `df` DataFrame only.
    2. Assign the final result (DataFrame, Series, or scalar) to a variable named `result`.
    3. Return your answer inside a single markdown fence: ```python ... ```
    """

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def ReasoningCurator(query: str, result: Any) -> str:
    """Builds the LLM prompt for reasoning about the result."""
    # Truncate result description to avoid overly long prompts
    if isinstance(result, (pd.DataFrame, pd.Series)):
        desc = str(result.head())[:300]
    else:
        desc = str(result)[:300]

    prompt = f"""
    The user asked: "{query}".
    The executed code produced the following result:
    ---
    {desc}
    ---
    Please provide a concise, 2-3 sentence explanation of what this result means in the context of the user's question.
    Focus on the insight, not the code or data structure.
    If the result is a plot, describe what the plot shows.
    """
    return prompt

def CodeGenerationAgent(query: str, df: pd.DataFrame) -> tuple[str, bool]:
    """Generates Python code based on the user's query."""
    should_plot = QueryUnderstandingTool(query)
    tool = PlotCodeGeneratorTool if should_plot else CodeWritingTool
    prompt = tool(df.columns.tolist(), query)

    system_prompt = """
    You are an expert Python data analyst. Write clean, efficient code to solve the user's request.
    Your response must contain ONLY a single, properly-formatted ```python code block.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0.1
    )
    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return code, should_plot

def ExecutionAgent(code: str, df: pd.DataFrame) -> Any:
    """Executes the generated code and returns the result."""
    if not code:
        return "Error: No code was generated to execute."
    env = {"pd": pd, "df": df.copy(), "plt": plt, "io": io}
    try:
        exec(code, {}, env)
        return env.get("result", "Error: Code executed but no 'result' variable was found.")
    except Exception as exc:
        return f"Error executing code: {exc}"

def ReasoningAgent(query: str, result: Any) -> str:
    """Generates a natural language explanation of the result."""
    prompt = ReasoningCurator(query, result)
    system_prompt = "You are an insightful data analyst who explains results clearly and concisely to non-technical users."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Generates a brief summary and exploratory questions for the dataset."""
    summary_prompt = f"""
    Analyze the following dataset summary:
    - Shape: {df.shape}
    - Columns: {df.columns.tolist()}
    - Data Types: {df.dtypes.to_dict()}
    - Missing Values: {df.isnull().sum().to_dict()}

    Provide a brief, one-paragraph description of what this dataset likely contains.
    Then, suggest 3 interesting data analysis questions that could be answered using this data.
    """
    messages = [
        {"role": "system", "content": "You are a helpful data analyst. Provide a concise summary and insightful questions."},
        {"role": "user", "content": summary_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"


# === Main Streamlit App ===============================================

def main():
    st.set_page_config(layout="wide", page_title="Data Analysis Agent")

    # Initialize session state variables
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    left, right = st.columns([0.3, 0.7])

    with left:
        st.header("Data Analysis Agent")
        # Correctly attribute the model being used in the code
        st.markdown("<small>Powered by **OpenAI GPT-4o**</small>", unsafe_allow_html=True)
        
        file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if file:
            # Load data and generate insights only when a new file is uploaded
            if "df" not in st.session_state or st.session_state.get("current_file") != file.name:
                with st.spinner("Loading data..."):
                    st.session_state.df = pd.read_csv(file)
                    st.session_state.current_file = file.name
                    st.session_state.messages = [] # Clear chat on new file
                with st.spinner("Generating dataset insights..."):
                    st.session_state.insights = DataInsightAgent(st.session_state.df)

            st.dataframe(st.session_state.df.head())
            with st.expander("Dataset Insights"):
                st.markdown(st.session_state.get("insights", "Could not generate insights."))
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data")
        
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        st.pyplot(st.session_state.plots[msg["plot_index"]], use_container_width=False)

        if file:
            if user_q := st.chat_input("Ask about your data..."):
                st.session_state.messages.append({"role": "user", "content": user_q})
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        code, should_plot = CodeGenerationAgent(user_q, st.session_state.df)
                        result_obj = ExecutionAgent(code, st.session_state.df)
                        explanation_text = ReasoningAgent(user_q, result_obj)

                    # Build and display the assistant's response
                    plot_idx = None
                    is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                    
                    if is_plot:
                        fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                        st.session_state.plots.append(fig)
                        plot_idx = len(st.session_state.plots) - 1

                    # Display the explanation first
                    st.markdown(explanation_text)

                    # Display the plot if one was created
                    if is_plot:
                        st.pyplot(st.session_state.plots[plot_idx], use_container_width=False)

                    # Show the generated code in a collapsible section
                    code_html = (
                        f'<details class="code"><summary>View generated code</summary>'
                        f'<pre><code class="language-python">{code}</code></pre></details>'
                    )
                    
                    # Store the complete message for history
                    assistant_msg = f"{explanation_text}\n\n{code_html}"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_msg,
                        "plot_index": plot_idx
                    })
                    # Use st.rerun() to ensure plots in history are displayed correctly
                    st.rerun()


if __name__ == "__main__":
    main()