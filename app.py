"""Streamlit UI for the IMDB conversational voice agent."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder

import agent as agent_mod
from config import CHAT_MODEL, CHROMA_DIR, EMBED_MODEL, OPENAI_API_KEY, STT_MODEL, TTS_MODEL, TTS_VOICE
from data.build_index import build as build_index
from data.loader import get_conn, load_df
from voice.stt import transcribe
from voice.tts import synthesize


st.set_page_config(page_title="Movie Voice Agent", page_icon="🎬", layout="wide")


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = agent_mod.initial_history()
    if "turns" not in st.session_state:
        st.session_state.turns = []  # list[dict]: {role, text, tables, recs, audio_mp3, clarification}
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
    if "pending_clarification" not in st.session_state:
        st.session_state.pending_clarification = None
    if "dataset_loaded" not in st.session_state:
        load_df()  # warm cache
        st.session_state.dataset_loaded = True


def _sidebar() -> None:
    with st.sidebar:
        st.header("🎬 Movie Voice Agent")
        st.caption("Conversational agent over the IMDB Top-1000 dataset")
        st.divider()

        key_ok = bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))
        st.markdown(
            f"**API key:** {'✅ loaded' if key_ok else '❌ missing (check .env)'}"
        )
        st.markdown(
            f"""
**Models**
- Chat: `{CHAT_MODEL}`
- Embeddings: `{EMBED_MODEL}`
- STT: `{STT_MODEL}`
- TTS: `{TTS_MODEL}` ({TTS_VOICE})
"""
        )

        st.divider()
        df = load_df()
        st.metric("Movies loaded", len(df))
        years = df["released_year"].dropna()
        if len(years):
            st.caption(f"Years: {int(years.min())} – {int(years.max())}")
        st.caption(f"Vector index: `{CHROMA_DIR.name}/`")

        st.divider()
        if st.button("🔄 Rebuild vector index", use_container_width=True):
            with st.spinner("Rebuilding Chroma index (≈ 30s)..."):
                build_index(force=True)
            st.success("Index rebuilt.")
            st.rerun()

        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.messages = agent_mod.initial_history()
            st.session_state.turns = []
            st.session_state.pending_clarification = None
            st.rerun()

        st.divider()
        with st.expander("Try these questions"):
            st.markdown(
                """
1. When did The Matrix release?
2. Top 5 movies of 2019 by meta score
3. Top 7 comedy movies between 2010–2020 by IMDB rating
4. Top horror movies with meta score above 85 and IMDB rating above 8
5. Top directors and their highest grossing movies with gross earnings > 500M at least twice
6. Top 10 movies with over 1M votes but lower gross earnings
7. Comedy movies where there is death or dead people involved
8. Summarize Steven Spielberg's top-rated sci-fi movie plots
9. Movies before 1990 that involve police in the plot
10. Al Pacino movies grossed over $50M with IMDB rating 8+
"""
            )


def _render_table(table: dict) -> None:
    rows = table.get("rows") or []
    if not rows:
        return
    df = pd.DataFrame(rows)
    display_df = df.drop(columns=["poster_link"], errors="ignore")
    if "overview" in display_df.columns:
        display_df["overview"] = display_df["overview"].astype(str).str.slice(0, 200) + "..."
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if "poster_link" in df.columns:
        posters = df.head(6)[["series_title", "poster_link"]].dropna()
        posters = posters[posters["poster_link"].str.startswith("http", na=False)]
        if len(posters):
            cols = st.columns(min(len(posters), 6))
            for col, (_, row) in zip(cols, posters.iterrows()):
                with col:
                    try:
                        st.image(row["poster_link"], caption=row["series_title"], use_container_width=True)
                    except Exception:
                        pass


def _render_recommendations(recs: list[dict]) -> None:
    if not recs:
        return
    st.markdown("**🍿 You might also like**")
    df = pd.DataFrame([
        {
            "Title": r.get("series_title"),
            "Year": r.get("released_year"),
            "Genre": r.get("genre"),
            "IMDB": r.get("imdb_rating"),
            "Meta": r.get("meta_score"),
            "Similarity": r.get("score"),
        }
        for r in recs
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        for table in turn.get("tables") or []:
            _render_table(table)
        if turn.get("recs"):
            _render_recommendations(turn["recs"])
        if turn.get("trace"):
            with st.expander("Reasoning trace"):
                for line in turn["trace"]:
                    st.code(line, language="text")
        if turn.get("audio_mp3"):
            st.audio(turn["audio_mp3"], format="audio/mp3", autoplay=turn.get("autoplay", False))


def _handle_user_message(user_text: str) -> None:
    if not user_text.strip():
        return
    st.session_state.turns.append({"role": "user", "text": user_text})

    with st.spinner("Thinking..."):
        response = agent_mod.run(st.session_state.messages, user_text)

    mp3 = b""
    if response.text and not response.clarification:
        with st.spinner("Generating voice reply..."):
            try:
                mp3 = synthesize(response.text)
            except Exception as e:
                st.warning(f"TTS failed: {e}")

    st.session_state.turns.append({
        "role": "assistant",
        "text": response.text or "(no reply)",
        "tables": response.tables,
        "recs": response.recommendations,
        "trace": response.reasoning_trace,
        "audio_mp3": mp3,
        "autoplay": True,
        "clarification": response.clarification,
    })

    if response.clarification:
        st.session_state.pending_clarification = response.clarification
    else:
        st.session_state.pending_clarification = None


def _clarification_panel() -> None:
    pending = st.session_state.pending_clarification
    if not pending:
        return
    with st.container(border=True):
        st.markdown(f"🤔 **{pending['question']}**")
        choice = st.radio(
            "Pick one:",
            pending["options"],
            key=f"clar_{len(st.session_state.turns)}",
            label_visibility="collapsed",
        )
        if st.button("Send answer", type="primary"):
            _handle_user_message(choice)
            st.rerun()


def _input_row() -> None:
    col_text, col_mic = st.columns([5, 1])

    with col_mic:
        audio = mic_recorder(
            start_prompt="🎙️",
            stop_prompt="⏹️",
            just_once=False,
            use_container_width=True,
            key="mic",
        )

    with col_text:
        user_text = st.chat_input("Ask about any movie, director, genre, or plot...")

    if audio and audio.get("id") and audio["id"] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio["id"]
        with st.spinner("Transcribing..."):
            try:
                transcribed = transcribe(audio["bytes"])
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                transcribed = ""
        if transcribed:
            _handle_user_message(transcribed)
            st.rerun()

    if user_text:
        _handle_user_message(user_text)
        st.rerun()


def main() -> None:
    _init_state()
    _sidebar()

    st.title("🎬 Movie Voice Agent")
    st.caption(
        "Ask about movies, directors, genres, or plot themes. Click the mic to speak, "
        "or type below. Answers include reasoning and similar-movie suggestions."
    )

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing. Add it to `.env` and restart.")
        st.stop()

    # Ensure DuckDB is ready
    get_conn()

    for turn in st.session_state.turns:
        _render_turn(turn)

    _clarification_panel()
    _input_row()


if __name__ == "__main__":
    main()
