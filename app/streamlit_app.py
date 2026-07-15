import os
import sys
import time
import tempfile

import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Add Project Root to Python Path
# ---------------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from inference.search import search_image

# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Product Matching Framework",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("📌 Project Details")

    st.markdown("### Model")
    st.write("Triplet Network + ResNet18")

    st.metric("Embedding Dimension", "128")
    st.metric("Dataset Images", "396 (Demo)")
    st.metric("Top Matches", "5")

    st.markdown("### Search Engine")
    st.write("FAISS")

    st.markdown("### Dataset")
    st.write("Stanford Online Products (Demo Subset)")

    st.divider()

    st.info(
        """
Upload any product image.

The framework extracts deep visual features and retrieves
the most visually similar products using FAISS similarity search.
"""
    )

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🛍️ Product Matching Framework")

st.markdown("""
### Deep Metric Learning using Triplet Loss + FAISS

Retrieve visually similar products using a **Triplet Network**
trained with **Triplet Loss** and accelerated using
**FAISS Approximate Nearest Neighbor Search**.
""")

st.info(
"""
This application generates image embeddings using a ResNet18-based
Triplet Network and performs efficient similarity search
over a demo subset of the Stanford Online Products dataset.
"""
)

st.divider()

# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a Product Image",
    type=["jpg", "jpeg", "png", "JPG"]
)

# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left_col, right_col = st.columns([1.2, 1.8])

    with left_col:

        st.subheader("📷 Query Image")

        st.image(
            image,
            use_container_width=True
        )

    search_button = st.button(
        "🔍 Search Similar Products",
        use_container_width=True
    )

    if search_button:

        temp_path = os.path.join(
            tempfile.gettempdir(),
            uploaded_file.name
        )

        image.save(temp_path)

        start_time = time.time()

        with st.spinner("Searching similar products..."):

            results = search_image(
                temp_path,
                top_k=5
            )

        end_time = time.time()

        medals = ["🥇", "🥈", "🥉", "🏅", "🏅"]

        with right_col:

            st.subheader("🎯 Top Similar Products")

            st.success(
                f"Retrieved {len(results)} products in {end_time-start_time:.2f} seconds"
            )

            for i, result in enumerate(results):

                with st.container(border=True):

                    st.subheader(
                        f"{medals[i]} Match #{i+1}"
                    )

                    img_col, info_col = st.columns([1, 2])

                    with img_col:

                        st.image(
                            result["image_path"],
                            use_container_width=True
                        )

                    with info_col:

                        similarity = result["score"] * 100

                        st.metric(
                            "Similarity Score",
                            f"{similarity:.2f}%"
                        )

                        st.progress(
                            min(similarity / 100, 1.0)
                        )

                        st.write("**Product ID**")
                        st.code(result["product_id"])
                        st.write("**Image Name**")
                        st.write(result["image_name"])

                        if similarity >= 99:

                            st.success("Excellent Match")

                        elif similarity >= 95:

                            st.info("Very Good Match")

                        elif similarity >= 90:

                            st.warning("Good Match")

                        else:

                            st.error("Low Similarity")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.markdown(
"""


**Technology Stack**

- PyTorch
- VGG16
- Triplet Loss
- FAISS
- Streamlit

**Dataset**

Stanford Online Products Dataset

---
*Built as a Deep Metric Learning based Product Image Retrieval System.*
"""
)