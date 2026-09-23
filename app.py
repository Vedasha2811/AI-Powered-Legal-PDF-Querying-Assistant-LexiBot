import streamlit as st
import pymupdf
import os

from dotenv import load_dotenv
from google import genai

from utils.chunker import create_chunks
from utils.rag import RAGSystem


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. "
        "Please create a .env file and add GEMINI_API_KEY."
    )
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LexiBot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# CREATE RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag_system():
    return RAGSystem()


rag_system = load_rag_system()


# ============================================================
# SESSION STATE
# ============================================================

if "summary" not in st.session_state:
    st.session_state.summary = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚖️ LexiBot")

    st.write(
        "AI-powered legal document analysis using "
        "Gemini and Retrieval-Augmented Generation (RAG)."
    )

    st.divider()

    st.subheader("Features")

    st.write("📄 PDF Text Extraction")
    st.write("🤖 AI Document Summary")
    st.write("🔍 Semantic Search")
    st.write("💬 Document Question Answering")

    st.divider()

    st.caption(
        "For informational purposes only. "
        "This application does not provide legal advice."
    )


# ============================================================
# MAIN UI
# ============================================================

st.title("⚖️ LexiBot")

st.markdown(
    """
    Upload a legal PDF document, generate an AI-powered summary,
    and ask questions about its contents using RAG.
    """
)

st.divider()


# ============================================================
# UPLOAD PDF
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # --------------------------------------------------------
    # OPEN PDF
    # --------------------------------------------------------

    document = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    # --------------------------------------------------------
    # CHECK EXTRACTED TEXT
    # --------------------------------------------------------

    if not text.strip():

        st.warning(
            "No readable text was found in this PDF. "
            "The PDF may contain scanned images instead of text."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE DOCUMENT CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        text,
        chunk_size=1000,
        overlap=200
    )

    # --------------------------------------------------------
    # BUILD FAISS INDEX
    # --------------------------------------------------------

    with st.spinner("Building document search index..."):

        rag_system.build_index(chunks)

    # --------------------------------------------------------
    # CREATE TABS
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📄 Document Text",
            "🤖 AI Summary",
            "💬 Ask Questions"
        ]
    )

    # ========================================================
    # TAB 1 - DOCUMENT TEXT
    # ========================================================

    with tab1:

        st.subheader("Extracted Document Text")

        st.text_area(
            "Document Content",
            text,
            height=500
        )

        st.caption(
            f"Document divided into {len(chunks)} chunks "
            "for AI processing."
        )

    # ========================================================
    # TAB 2 - AI SUMMARY
    # ========================================================

    with tab2:

        st.subheader("AI Document Summary")

        if st.button(
            "Generate Summary",
            type="primary"
        ):

            chunk_summaries = []

            # ------------------------------------------------
            # SUMMARIZE EACH CHUNK
            # ------------------------------------------------

            for i, chunk in enumerate(chunks):

                with st.spinner(
                    f"Analyzing chunk {i + 1} of {len(chunks)}..."
                ):

                    prompt = f"""
You are an AI assistant specialized in analyzing legal documents.

Summarize the following section of a legal document.

Rules:

- Only use information present in the text.
- Do not invent facts.
- Preserve important legal details.
- Identify parties, dates, financial terms,
  obligations, rights, restrictions, and important clauses.
- If something is not mentioned, do not assume it.
- Keep the summary concise.

DOCUMENT SECTION:

{chunk}
"""

                    try:

                        response = client.models.generate_content(
                            model="gemini-3.5-flash",
                            contents=prompt
                        )

                        if response.text:
                            chunk_summaries.append(
                                response.text
                            )

                    except Exception as e:

                        st.error(
                            f"Error analyzing chunk {i + 1}: {e}"
                        )

                        st.stop()

            # ------------------------------------------------
            # CHECK CHUNK SUMMARIES
            # ------------------------------------------------

            if not chunk_summaries:

                st.error(
                    "Unable to generate summaries for the document."
                )

                st.stop()

            # ------------------------------------------------
            # COMBINE CHUNK SUMMARIES
            # ------------------------------------------------

            combined_summary = "\n\n".join(
                chunk_summaries
            )

            # ------------------------------------------------
            # GENERATE FINAL SUMMARY
            # ------------------------------------------------

            with st.spinner(
                "Creating final summary..."
            ):

                final_prompt = f"""
You are an AI assistant specialized in legal document analysis.

Below are summaries of different sections of the same
legal document.

Combine them into one clear and structured final summary.

Do not add information that is not present in the summaries.

Use the following structure:

## 1. Document Overview

## 2. Parties Involved

## 3. Important Dates

## 4. Financial Terms

## 5. Responsibilities of Each Party

## 6. Key Clauses

## 7. Termination Conditions

## 8. Potential Risks / Important Points

## 9. Simple Summary

SECTION SUMMARIES:

{combined_summary}
"""

                try:

                    final_response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=final_prompt
                    )

                    if final_response.text:

                        st.session_state.summary = (
                            final_response.text
                        )

                    else:

                        st.error(
                            "Gemini returned an empty response."
                        )

                except Exception as e:

                    st.error(
                        f"Error generating final summary: {e}"
                    )

        # ----------------------------------------------------
        # DISPLAY SUMMARY
        # ----------------------------------------------------

        if st.session_state.summary:

            st.markdown(
                st.session_state.summary
            )

            st.download_button(
                label="⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="legal_document_summary.txt",
                mime="text/plain"
            )

            st.info(
                "Disclaimer: This summary is AI-generated "
                "and is for informational purposes only. "
                "It is not legal advice."
            )

        else:

            st.write(
                "Click **Generate Summary** to analyze "
                "the document."
            )

    # ========================================================
    # TAB 3 - ASK QUESTIONS
    # ========================================================

    with tab3:

        st.subheader(
            "Ask Questions About the Document"
        )

        question = st.text_input(
            "Enter your question:",
            placeholder=(
                "Example: What is the termination "
                "notice period?"
            )
        )

        if st.button(
            "Ask Question",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                # ------------------------------------------------
                # FIND RELEVANT CHUNKS
                # ------------------------------------------------

                with st.spinner(
                    "Finding relevant information..."
                ):

                    try:

                        relevant_results = rag_system.search(
                            question,
                            k=3
                        )

                    except Exception as e:

                        st.error(
                            f"Error searching the document: {e}"
                        )

                        st.stop()

                    if not relevant_results:

                        st.warning(
                            "No relevant information was found "
                            "in the document."
                        )

                        st.stop()

                    relevant_chunks = [
                        result["text"]
                        for result in relevant_results
                    ]

                    context = "\n\n".join(
                        relevant_chunks
                    )

                # ------------------------------------------------
                # CREATE RAG PROMPT
                # ------------------------------------------------

                question_prompt = f"""
You are an AI assistant specialized in legal document analysis.

Answer the user's question using ONLY the provided
document context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context,
   clearly say that the information is not available
   in the document.
3. Give a concise and accurate answer.
4. Preserve the legal meaning.
5. Do not provide legal advice.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

                # ------------------------------------------------
                # GENERATE ANSWER
                # ------------------------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    try:

                        response = client.models.generate_content(
                            model="gemini-3.5-flash",
                            contents=question_prompt
                        )

                    except Exception as e:

                        st.error(
                            f"Error generating answer: {e}"
                        )

                        st.stop()

                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.subheader("AI Answer")

                if response.text:

                    st.markdown(
                        response.text
                    )

                else:

                    st.warning(
                        "Gemini returned an empty response."
                    )

                st.caption(
                    "Answer generated from the uploaded "
                    "document using RAG."
                )
