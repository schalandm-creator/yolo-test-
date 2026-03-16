# app.py

import streamlit as st

# ────────────────────────────────────────────────
# Streamlit zuerst komplett initialisieren
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fundbüro – YOLO Erkennung",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧥 Fundbüro – lokale YOLO Erkennung")
st.caption("2026 – ohne Datenbank – nur Browser-Session")

st.info("""
Diese Version lädt YOLO **erst nach** dem Streamlit-Setup,  
um den bekannten Streamlit-Watcher + Torch-Konflikt zu vermeiden.
""")

# ────────────────────────────────────────────────
# Jetzt erst Torch / ultralytics / cv2 laden
# ────────────────────────────────────────────────
with st.spinner("Lade YOLO-Modell (einmalig) ..."):
    try:
        from ultralytics import YOLO
        from PIL import Image
        import io
        import uuid
        from datetime import datetime

        @st.cache_resource(show_spinner=False)
        def load_yolo():
            # yolo11n = schnell & leicht, yolo11s = besser genau
            return YOLO("yolo11n.pt")

        model = load_yolo()
        st.success("YOLO erfolgreich geladen", icon="✅")

    except Exception as e:
        st.error(f"Laden fehlgeschlagen\n\n{e}")
        st.stop()

# ────────────────────────────────────────────────
# Session State
# ────────────────────────────────────────────────
if "fund_items" not in st.session_state:
    st.session_state.fund_items = []

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Hochladen & Analysieren", "🖼️ Galerie"])

# ────────────────────────────────────────────────
# Tab 1 – Upload
# ────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Bild hochladen (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        bytes_data = uploaded.read()
        img = Image.open(io.BytesIO(bytes_data))

        st.image(img, caption="Originalbild", use_container_width=True)

        with st.spinner("Analyse läuft ..."):
            try:
                results = model(img, conf=0.35, iou=0.45)
                annotated = results[0].plot() if results[0].boxes else img

                st.image(annotated, caption="YOLO-Ergebnis", use_container_width=True)

                classes = [model.names[int(box.cls)] for box in results[0].boxes]
                unique_classes = list(dict.fromkeys(classes))

                if unique_classes:
                    st.success(f"Erkannt: **{', '.join(unique_classes)}**")
                else:
                    st.warning("Keine Objekte mit ausreichender Sicherheit gefunden.")

                notes = st.text_input("Notizen / Beschreibung", "")

                if st.button("Als Fundstück speichern"):
                    entry = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded.name,
                        "image_bytes": bytes_data,
                        "classes": unique_classes,
                        "notes": notes.strip() or "—"
                    }
                    st.session_state.fund_items.append(entry)
                    st.success("Gespeichert (nur in diesem Browser sichtbar)")
                    st.balloons()

            except Exception as e:
                st.error(f"Analyse-Fehler\n\n{e}")

# ────────────────────────────────────────────────
# Tab 2 – Galerie + Filter
# ────────────────────────────────────────────────
with tab2:
    st.subheader(f"Galerie – {len(st.session_state.fund_items)} Einträge")

    if not st.session_state.fund_items:
        st.info("Noch keine Fundstücke vorhanden.")
    else:
        # Filter-Optionen
        all_classes = set()
        for item in st.session_state.fund_items:
            all_classes.update(item["classes"])

        selected = st.multiselect(
            "Nach Kategorie filtern",
            sorted(all_classes),
            default=list(all_classes)
        )

        filtered = [
            it for it in st.session_state.fund_items
            if not selected or any(c in it["classes"] for c in selected)
        ]

        cols = st.columns(3)
        for i, item in enumerate(filtered):
            col = cols[i % 3]
            img = Image.open(io.BytesIO(item["image_bytes"]))
            col.image(img, use_column_width=True)

            txt = ", ".join(item["classes"]) or "keine Erkennung"
            col.markdown(f"**{txt}**")
            if item["notes"]:
                col.caption(item["notes"])
            col.caption(item["timestamp"])

    if st.button("Alles löschen"):
        st.session_state.fund_items = []
        st.rerun()

st.divider()
st.caption("Lokale Version – Daten nur im Browser (session_state)")
