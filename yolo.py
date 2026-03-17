import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="KI Bilderkennung", layout="wide")

# ─── Modell-Cache ───────────────────────────────────────
@st.cache_resource(show_spinner="Lade Vision Transformer …")
def load_classifier():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_classifier()

# ─── Session State ──────────────────────────────────────
if "results_history" not in st.session_state:
    st.session_state.results_history = []

# ─── Header & Upload ────────────────────────────────────
st.title("🖼️ KI Bilderkennung – Vision Transformer")
st.caption("ViT-base • ImageNet-1k • Top-5 Ergebnisse")

cols = st.columns([3, 1])
with cols[0]:
    uploaded_files = st.file_uploader(
        "Bild(er) hochladen … (mehrere möglich)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

# ─── Verarbeitung ───────────────────────────────────────
if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)

        with st.spinner(f"Analysiere {file.name} …"):
            try:
                results = classifier(image)
                top5 = results[:5]

                # Für Tabelle & History
                row = {
                    "Dateiname": file.name,
                    "Top 1": top5[0]["label"],
                    "Sicherheit Top1": f"{top5[0]['score']:.1%}",
                    "Top 5": " | ".join([f"{r['label']} ({r['score']:.0%})" for r in top5]),
                    "Bild": image
                }

                st.session_state.results_history.append(row)

            except Exception as e:
                st.error(f"Fehler bei {file.name}: {e}")

# ─── Ergebnisse anzeigen ────────────────────────────────
if st.session_state.results_history:
    df = pd.DataFrame(st.session_state.results_history)

    tab1, tab2 = st.tabs(["📊 Tabelle", "🖼️ Galerie"])

    with tab1:
        st.dataframe(
            df[["Dateiname", "Top 1", "Sicherheit Top1", "Top 5"]],
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        cols = st.columns(3)
        for i, row in enumerate(df.itertuples()):
            col = cols[i % 3]
            col.image(row.Bild, use_column_width=True)
            col.markdown(f"**{row.Dateiname}**")
            col.markdown(f"**Top 1:** {row._2} ({row._3})")

    if st.button("Verlauf löschen"):
        st.session_state.results_history = []
        st.rerun()

else:
    st.info("Noch keine Bilder analysiert. ↑ Lade welche hoch …")
