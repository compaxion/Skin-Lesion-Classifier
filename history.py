"""
history.py
Renders user prediction history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from db import delete_prediction, get_prediction, list_predictions_for_user


def render_history_page(user: dict) -> None:
    st.title("My History")
    st.caption(f"Signed in as **{user['username']}**")
    st.markdown("---")

    rows = list_predictions_for_user(user["id"])

    if not rows:
        st.info("You have no past uploads yet. Run a classification "
                "from the **Classifier** page and it will appear here.")
        return

    # Build summary table
    summary = pd.DataFrame(
        [
            {
                "ID":         r["id"],
                "When":       r["created_at"],
                "File":       r["original_filename"],
                "Diagnosis":  r["predicted_class"],
                "Confidence": f"{r['confidence']:.2f}%",
            }
            for r in rows
        ]
    )
    st.markdown(summary.to_html(index=False, escape=True, classes="themed-table"),
                unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Inspect a record")

    ids = [r["id"] for r in rows]
    selected_id = st.selectbox(
        "Choose a record to view in detail:",
        options=ids,
        format_func=lambda i: f"#{i} — {next(r['original_filename'] for r in rows if r['id']==i)}",
        key="history_selector",
    )

    # Re-query to enforce user_id filter
    record = get_prediction(user["id"], selected_id)
    if record is None:
        st.error("Record not found.")
        return

    col_img, col_info = st.columns([1, 1])

    with col_img:
        image_path = Path(record["image_path"])
        if image_path.is_file():
            st.image(str(image_path), caption=record["original_filename"],
                     use_container_width=True)
        else:
            st.warning("The original image file is missing from disk.")

    with col_info:
        st.metric("Predicted Diagnosis", record["predicted_class"])
        st.metric("Confidence", f"{record['confidence']:.2f}%")
        st.caption(f"Uploaded: {record['created_at']}")

        try:
            probs = json.loads(record["probabilities_json"])
            prob_df = (
                pd.DataFrame(
                    [{"Diagnosis": k, "Probability": v * 100} for k, v in probs.items()]
                )
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            prob_df["Probability"] = prob_df["Probability"].map(lambda x: f"{x:.2f}%")
            st.markdown(prob_df.to_html(index=False, escape=True, classes="themed-table"),
                        unsafe_allow_html=True)
        except (json.JSONDecodeError, TypeError):
            st.caption("Stored probabilities are unreadable.")

    # Delete confirmation
    st.markdown("---")
    with st.expander("Delete this record"):
        st.warning("This will permanently remove the record and the "
                   "stored image file. This cannot be undone.")
        confirm = st.checkbox("Yes, I'm sure.", key=f"confirm_del_{selected_id}")
        if st.button("Delete record", disabled=not confirm, key=f"del_btn_{selected_id}"):
            # Remove image file, then DB record
            try:
                image_path = Path(record["image_path"])
                if image_path.is_file():
                    image_path.unlink()
            except OSError as exc:
                st.warning(f"Could not remove image file: {exc}")

            if delete_prediction(user["id"], selected_id):
                st.success("Record deleted.")
                st.rerun()
            else:
                st.error("Could not delete the record.")
