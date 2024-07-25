import streamlit as st
from streamlit.components.v1 import html
from pystac_client import Client
from dateutil import tz
from datetime import date, datetime
from streamlit_folium import st_folium
import folium

import planetary_computer

from utils.sentinel_analysis import (
    sentinel_2_query,
    sentinel_2_band_process,
    sentinel_1_query,
    sentinel_1_band_process,
)
from utils.wizard_decorators import next_step, previous_step, start_again

## See source at https://odc-stac.readthedocs.io/en/latest/notebooks/stac-load-e84-aws.html
# DATA_URL = "https://earth-search.aws.element84.com/v1"
DATA_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

## Details on Sentinel L1C at
COLLECTIONS = [
    "sentinel-2-l2a",  # Imagery: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/
    "sentinel-1-rtc",  # Radar
]  # , "sentinel-2-l1c", "landsat-c2-l2"]

## Steps
STEP_LIST = ["Select area", "Explore data", "Combine bands"]


def set_tz(date: date):
    """Set the timezone of a datetime.date object"""
    return datetime(date.year, date.month, date.day, tzinfo=tz.tzlocal())


def main():
    ## -----------------------------
    ## Initial configuration
    ## -----------------------------
    st.set_page_config(page_icon="🛰️", page_title="Water and remote sensing")

    with open("assets/stylesheet.css", "r") as f:
        st.html(f"<style>{f.read()}</style>")

    with open("assets/script.js", "r") as f:
        html(f"<script>{f.read()}</script>", height=0)

    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    if "catalog" not in st.session_state:
        st.session_state.catalog = Client.open(
            DATA_URL,
            modifier=planetary_computer.sign_inplace,
        )

    header_col, nav_col = st.columns([5, 1], vertical_alignment="bottom")

    with header_col:
        st.header("🛰️ Water identification using remote sensing data", anchor=False)

    with nav_col:
        st.caption(
            "<p style='text-align: right; line-height: 1rem;'>\n"
            f"Step {st.session_state.wizard_step + 1} of {len(STEP_LIST)}"
            f"<br><strong>{STEP_LIST[st.session_state.wizard_step]}</strong></p>",
            unsafe_allow_html=True,
        )

    ## -----------------------------
    ## First step - Map interaction
    ## -----------------------------
    if st.session_state.wizard_step == 0:
        with st.form("Map selection"):
            tools_col, map_col = st.columns((1, 2))
            with tools_col:
                date_range_placeholder = st.empty()
                collection_placeholder = st.empty()
                cloud_cover_placeholder = st.empty()

                submit_button = st.form_submit_button(
                    ":blue[**Retrieve data** ⭢]",
                    use_container_width=True,
                )

        with date_range_placeholder:
            naive_date_range = st.date_input(
                "Date range",
                (
                    datetime(2024, 1, 1, tzinfo=tz.tzlocal()),
                    datetime(2024, 7, 15, tzinfo=tz.tzlocal()),
                ),
            )

            st.session_state.date_range = [set_tz(t) for t in naive_date_range]

        with collection_placeholder:
            st.session_state.collection = st.selectbox(
                "Collection", COLLECTIONS, index=0
            )

        with cloud_cover_placeholder:
            if st.session_state.collection == "sentinel-2-l2a":
                st.session_state.cloud_cover = st.slider(
                    "Cloud coverage (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 15),
                    help="""Cloud coverage is calculated for a complete image.
                        \n\nImages with high cloud coverage can still have clear 
                        views of the area of interest.""",
                )

        with map_col:
            if "stfolium_kwargs" not in st.session_state:
                st.session_state.stfolium_kwargs = {
                    "center": (34.6037, -84.6852),
                    "zoom": 14,
                }

            current_map_view = st_folium(
                folium.Map(),
                use_container_width=True,
                height=390,
                returned_objects=["center", "zoom", "bounds"],
                **st.session_state.stfolium_kwargs,
            )

        if submit_button:
            if current_map_view["zoom"] < 13:
                st.toast("Zoom in the map to narrow down the query.", icon="🔎")

            else:
                folium_bounds = current_map_view["bounds"]
                st.session_state.bbox = [
                    folium_bounds["_southWest"]["lng"],  # low-left axis 1
                    folium_bounds["_southWest"]["lat"],  # low-left axis 2
                    folium_bounds["_northEast"]["lng"],  # up-right axis 1
                    folium_bounds["_northEast"]["lat"],  # up-right axis 2
                ]

                st.session_state.stfolium_kwargs = {
                    "center": (
                        current_map_view["center"]["lat"],
                        current_map_view["center"]["lng"],
                    ),
                    "zoom": current_map_view["zoom"],
                }

                st.session_state.wizard_step += 1
                st.rerun()

    ## -----------------------------
    ## Second step - Bands explore
    ## -----------------------------
    elif st.session_state.wizard_step == 1:
        if st.session_state.collection == "sentinel-2-l2a":
            n_matches = sentinel_2_query()

        elif st.session_state.collection == "sentinel-1-rtc":
            n_matches = sentinel_1_query()

        if not n_matches:
            st.error(
                "No items were found... go back and try modifying your query.",
                icon=":material/data_alert:",
            )

        buttons_cols = st.columns([1, 2.5])

        with buttons_cols[1]:
            st.button(
                ":blue[**Calculate bands** ⭢]",
                use_container_width=True,
                on_click=next_step,
                disabled=not n_matches,
            )

        with buttons_cols[0]:
            st.button(
                ":gray[⭠ Go back]",
                use_container_width=True,
                on_click=previous_step,
            )

    ## -----------------------------
    ## Third step - Bands processing
    ## -----------------------------
    elif st.session_state.wizard_step == 2:
        if st.session_state.collection == "sentinel-2-l2a":
            sentinel_2_band_process()

        elif st.session_state.collection == "sentinel-1-rtc":
            sentinel_1_band_process()

        buttons_cols = st.columns([1, 2.5])

        with buttons_cols[0]:
            st.button(
                ":gray[⭠ Go back]",
                use_container_width=True,
                on_click=previous_step,
            )

        with buttons_cols[1]:
            st.button(
                ":blue[Start again ⭯]",
                use_container_width=True,
                on_click=start_again,
            )


if __name__ == "__main__":
    main()
