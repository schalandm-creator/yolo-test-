import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import uuid
from datetime import datetime

st.set_page_config(page_title="Fundbüro – lokal mit YOLO", layout="wide")
st.title("🧥 Fundbüro (lokale Version – ohne Datenbank)")

st.info("Alle Fundstücke werden nur im Browser gespeichert (session_state). Beim Neustart oder Schließen des Tabs sind sie weg.")

# ────────────────────────────────────────────────
# YOLO Modell laden (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO("yolo11n.pt")   # oder "yolo11s.pt" / "yolo12n.pt" – nano ist am schnellsten

model = load_yolo_model()

# ────────────────────────────────────────────────
# Session State initialisieren
# ────────────────────────────────────────────────
if "fund_items" not in st.session_state:
    st.session_state.fund_items = []

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Neues Fundstück hochladen", "🖼️ Galerie"])

# ────────────────────────────────────────────────
# TAB 1: Hochladen & Analysieren
# ────────────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader("Bild des Fundstücks hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Bild anzeigen
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        # YOLO Analyse
        with st.spinner("YOLO erkennt Gegenstände ..."):
            results = model(image, conf=0.35, iou=0.45)
            annotated_img = results[0].plot()  # Bild mit Bounding Boxes

            st.image(annotated_img, caption="YOLO-Erkennung", use_container_width=True)

            # Erkannte Klassen sammeln
            detected = []
            for box in results[0].boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf = float(box.conf)
                detected.append({"class": label, "confidence": conf})

            if detected:
                st.success(f"{len(detected)} Gegenstände erkannt")
                for item in detected:
                    st.write(f"• {item['class']} ({item['confidence']:.1%})")
            else:
                st.warning("Keine Gegenstände mit ausreichender Sicherheit erkannt.")

            # Notizen
            notes = st.text_input("Zusätzliche Notizen / Beschreibung", "")

            if st.button("💾 Als Fundstück speichern"):
                # Bild als Bytes speichern (für Galerie)
                item_id = str(uuid.uuid4())
                entry = {
                    "id": item_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "original_name": uploaded_file.name,
                    "image_bytes": image_bytes,          # ← Bilddaten im RAM
                    "detected_classes": [d["class"] for d in detected],
                    "notes": notes.strip() or "Keine Notiz"
                }

                st.session_state.fund_items.append(entry)
                st.success("Fundstück gespeichert! (nur in diesem Browser-Tab sichtbar)")
                st.balloons()

# ────────────────────────────────────────────────
# TAB 2: Galerie + Filter
# ────────────────────────────────────────────────
with tab2:
    st.subheader("Deine Fundstücke-Galerie")

    if not st.session_state.fund_items:
        st.info("Noch keine Fundstücke gespeichert.")
    else:
        # Alle Klassen sammeln für Filter
        all_classes = set()
        for item in st.session_state.fund_items:
            all_classes.update(item["detected_classes"])
        all_classes = sorted(list(all_classes))

        # Filter
        selected_classes = st.multiselect(
            "Nach Kategorie filtern",
            options=all_classes,
            default=all_classes
        )

        filtered_items = [
            item for item in st.session_state.fund_items
            if not selected_classes or any(c in item["detected_classes"] for c in selected_classes)
        ]

        if not filtered_items:
            st.warning("Keine Fundstücke passen zum Filter.")
        else:
            cols = st.columns(3)
            for i, item in enumerate(filtered_items):
                col = cols[i % 3]
                img = Image.open(io.BytesIO(item["image_bytes"]))
                col.image(img, use_column_width=True)
                col.markdown(f"**{', '.join(item['detected_classes']) or 'unbekannt'}**")
                if item["notes"]:
                    col.caption(item["notes"])
                col.caption(item["timestamp"])

    # Alles löschen Button (für Tests)
    if st.button("🗑️ Alle Fundstücke löschen (Session zurücksetzen)"):
        st.session_state.fund_items = []
        st.rerun()
