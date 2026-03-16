import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import uuid
from datetime import datetime

# ────────────────────────────────────────────────
# Seiten-Konfiguration
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fundbüro – YOLO Erkennung",
    page_icon="🧥",
    layout="wide"
)

st.title("🧥 Fundbüro – KI Erkennung mit YOLO")
st.caption("Bilder hochladen → YOLO erkennt Gegenstände → Galerie im Browser (Session)")

st.info("Diese App speichert **nichts** dauerhaft. Alle Daten verschwinden, sobald du den Tab schließt oder die Seite neu lädst.")

# ────────────────────────────────────────────────
# YOLO Modell einmal laden (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def get_yolo_model():
    try:
        return YOLO("yolo11n.pt")          # nano – schnell & leicht
        # Alternativen: "yolo11s.pt", "yolo12n.pt"
    except Exception as e:
        st.error(f"Fehler beim Laden des YOLO-Modells:\n{e}")
        st.stop()

model = get_yolo_model()

# ────────────────────────────────────────────────
# Session State für Fundstücke
# ────────────────────────────────────────────────
if "items" not in st.session_state:
    st.session_state.items = []

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab_upload, tab_gallery = st.tabs(["📸 Neues Fundstück", "🖼️ Galerie"])

# ────────────────────────────────────────────────
# Tab 1: Hochladen & Analysieren
# ────────────────────────────────────────────────
with tab_upload:
    uploaded_file = st.file_uploader(
        "Bild hochladen (jpg, png, jpeg)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Bild laden & anzeigen
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        # YOLO Analyse
        with st.spinner("YOLO analysiert das Bild ..."):
            try:
                results = model(image, conf=0.35, iou=0.45)
                annotated = results[0].plot() if results[0].boxes else image

                st.image(annotated, caption="YOLO-Erkennung", use_container_width=True)

                # Erkannte Klassen sammeln
                detected_classes = []
                confidences = []

                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    label = model.names[cls_id]
                    conf = float(box.conf)
                    detected_classes.append(label)
                    confidences.append(conf)

                # Duplikate entfernen, aber Häufigkeit beibehalten
                unique_classes = list(dict.fromkeys(detected_classes))

                if detected_classes:
                    st.success(f"Erkannt: **{', '.join(unique_classes)}**")
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    st.write(f"Durchschnittliche Sicherheit: **{avg_conf:.1%}**")
                else:
                    st.warning("Keine Objekte mit ausreichender Sicherheit erkannt.")

                # Notizen
                notes = st.text_input("Zusätzliche Notizen (z. B. Farbe, Zustand, Fundort)", "")

                if st.button("💾 Als Fundstück speichern"):
                    entry = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "image_bytes": image_bytes,
                        "detected_classes": unique_classes,
                        "avg_confidence": avg_conf,
                        "notes": notes.strip() or "Keine Notiz"
                    }
                    st.session_state.items.append(entry)
                    st.success("Gespeichert! (nur in diesem Browser sichtbar)")
                    st.balloons()

            except Exception as e:
                st.error(f"Fehler bei der YOLO-Analyse:\n{e}")

# ────────────────────────────────────────────────
# Tab 2: Galerie + Filter
# ────────────────────────────────────────────────
with tab_gallery:
    st.subheader(f"Galerie ({len(st.session_state.items)} Fundstücke)")

    if not st.session_state.items:
        st.info("Noch keine Fundstücke gespeichert.")
    else:
        # Alle Klassen für Filter sammeln
        all_classes = set()
        for item in st.session_state.items:
            all_classes.update(item["detected_classes"])
        all_classes = sorted(list(all_classes))

        selected = st.multiselect(
            "Nach erkannten Gegenständen filtern",
            options=all_classes,
            default=all_classes
        )

        filtered = [
            item for item in st.session_state.items
            if not selected or any(c in item["detected_classes"] for c in selected)
        ]

        if not filtered:
            st.warning("Keine Fundstücke passen zum Filter.")
        else:
            cols = st.columns(3)
            for i, item in enumerate(filtered):
                col = cols[i % 3]
                img = Image.open(io.BytesIO(item["image_bytes"]))
                col.image(img, use_column_width=True)

                classes_str = ", ".join(item["detected_classes"]) or "keine Erkennung"
                col.markdown(f"**{classes_str}** ({item['avg_confidence']:.1%})")
                if item["notes"]:
                    col.caption(item["notes"])
                col.caption(item["timestamp"])

    # Reset-Button
    if st.button("🗑️ Alle Fundstücke löschen"):
        st.session_state.items = []
        st.rerun()
