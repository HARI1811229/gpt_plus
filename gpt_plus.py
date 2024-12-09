import streamlit as st
from langchain.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
import wikipedia
search = DuckDuckGoSearchRun()

def get_ollama_model():
    return Ollama(model="qwen2.5:0.5b",num_ctx=12000,num_gpu=-1,temperature=0)

def duckduckgo_search(query):
    try:
        results = search.invoke(query)
        return [results]
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

def wikipedia_summary(query):
    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return {"title": query, "snippet": summary, "link": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"}
    except Exception as e:
        print(f"Wikipedia search error: {e}")
        return None

def generate_prompt(user_query, search_results, history):
    # print("search_results :",search_results)
    search_context = search_results
    wiki_summary = wikipedia_summary(user_query)
    if wiki_summary:
        search_context += f"\n\nWikipedia Summary: {wiki_summary}"
    print("search_context :", search_context)
    history_context = "\n\n".join(history[-5:] if len(history) > 5 else history)
    return f"""
    Conversation History:\n{history_context}
    User Query: {user_query}
    Search Results:\n{search_context}
    Based on the above information, provide an accurate and detailed response to the query. If the search results do not contain sufficient information, respond to the best of your ability based on your general knowledge.
    """
def main():
    st.set_page_config(page_title="GPT", page_icon="🤖", layout="wide")
    st.title("GPT with DuckDuckGo & Wikipedia 🦆📚")
    if "history" not in st.session_state:
        st.session_state.history = []
    with st.sidebar:
        st.header("About GPT")
        st.write(
            "This application combines the power of DuckDuckGo and Wikipedia with an advanced language model to provide accurate and detailed responses.")
        st.markdown("---")
        st.write("💡 **How to use:**")
        st.write(
            "1. Enter your query in the text box.\n2. Click **Get Answer** to fetch results and generate a response.")
        st.write("3. View the conversation history below.")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🔍 Enter your question below:")
        user_query = st.text_input("", "", placeholder="Type your query here...")

        if st.button("Get Answer") and user_query:
            with st.spinner("Fetching search results..."):
                duckduckgo_results = duckduckgo_search(user_query)
                print("duckduckgo_results : ",duckduckgo_results)
                wiki_summary = wikipedia_summary(user_query)
                print("wiki_summary : ",wiki_summary)
                search_results = duckduckgo_results
                if wiki_summary:
                    search_results.append(wiki_summary)
            prompt = generate_prompt(user_query, search_results, st.session_state.history)
            print("prompt : ",prompt)
            with st.spinner("Generating response..."):
                llm = get_ollama_model()
                response = llm(prompt)
            st.session_state.history.append(f"User Query: {user_query}\nModel Response: {response}")
            st.subheader("🤖 Response:")
            st.success(response)
    with col2:
        st.subheader("📝 Conversation History")
        if st.session_state.history:
            for entry in st.session_state.history[-5:]:
                st.text_area("", entry, height=150, disabled=True)
        else:
            st.info("No conversation history yet. Start by asking a question!")
if __name__ == "__main__":
    main()