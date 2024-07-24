import numpy as np
import xarray as xr
import streamlit as st
import pandas as pd
from odc.stac import stac_load
from numpy import datetime_as_string
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum


def normalize(array: np.ndarray):
    lower, upper = np.percentile(array.flatten(), [0, 99])
    return (array - lower) / (upper - lower)


def rgb_from_radar(single: xr.DataArray):
    vv = np.log10(single["vv"].to_numpy())
    vh = np.log10(single["vh"].to_numpy())
    ratio = (single["vh"] / single["vv"]).to_numpy()

    vvvh = normalize(np.dstack((vv, vh)))
    norm_ratio = normalize(ratio)

    return np.dstack((vvvh, norm_ratio))


def band_calculations(single: xr.DataArray):
    red = single["B04"].to_numpy()
    green = single["B03"].to_numpy()
    blue = single["B02"].to_numpy()
    nir = single["B08"].to_numpy()
    swir16 = single["B11"].to_numpy()

    rgb = normalize(np.dstack((red, green, blue)))
    fake = normalize(np.dstack((nir, red, green)))
    land_water = normalize(np.dstack((nir, swir16, red)))

    ndwi = (normalize(green) - normalize(nir)) / (normalize(green) + normalize(nir))

    return rgb, fake, land_water, ndwi


def sentinel_2_query():
    collection_details = st.session_state.catalog.get_collection(
        st.session_state.collection
    )

    band_details = pd.DataFrame(collection_details.summaries.get_list("eo:bands"))
    band_dict = band_details.set_index("name").to_dict("index")

    query = st.session_state.catalog.search(
        collections=[st.session_state.collection],
        bbox=st.session_state.bbox,
        datetime=st.session_state.date_range,
        query=[
            f"eo:cloud_cover>{st.session_state.cloud_cover[0]}",
            f"eo:cloud_cover<{st.session_state.cloud_cover[1]}",
        ],
    )

    items = list(query.items())
    n_matches = len(items)

    st.markdown(f":green-background[**{n_matches}** items were found.]")

    if n_matches:
        with st.spinner("Loading data..."):
            data = stac_load(
                items,
                bbox=st.session_state.bbox,
                groupby="solar_day",
                chunks={},  # <-- use Dask
            )

        tools_cols = st.columns([3, 1])
        image_placeholder = st.empty()

        with st.expander(
            f"See the bands' description in the **`{st.session_state.collection}`** collection"
        ):
            st.dataframe(
                band_details[["name", "description", "gsd"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "name": st.column_config.TextColumn("Name"),
                    "description": st.column_config.TextColumn("Description"),
                    "gsd": st.column_config.NumberColumn("Resolution", format="%d m"),
                },
            )

        with st.expander("See the **`xarray`** dataset"):
            st.html(data._repr_html_())

        with tools_cols[0]:
            band = st.selectbox(
                "Bands Available",
                options=band_dict.keys(),
                format_func=lambda x: band_dict[x]["description"].title() + f" ({x})",
                index=8,
            )

        with tools_cols[1]:
            timestamp = st.selectbox(
                "Date Available",
                options=data.time.data,
                format_func=lambda x: str(x).partition("T")[0],
            )

        with image_placeholder:
            with st.spinner("Drawing images..."):
                fig, ax = plt.subplots(figsize=[6, 4.8])
                data[band].sel(time=timestamp).plot.imshow(
                    ax=ax, cmap="Spectral_r", cbar_kwargs=dict(shrink=0.5)
                )
                ax.set_aspect("equal")
                ax.set_title(
                    datetime_as_string(timestamp, unit="m").replace("T", " | ")
                )
                ax.xaxis.label.set_visible(False)
                ax.yaxis.label.set_visible(False)
                ax.tick_params(labelsize=4)
                ax.tick_params(axis="x", labelrotation=90)
                ax.ticklabel_format(useOffset=False, style="plain")

                fig_hist, ax = plt.subplots(figsize=[6, 4])
                ax.hist(
                    data[band].sel(time=timestamp).to_numpy().flatten(),
                    bins="scott",
                    color="#43a2ca",
                )
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.set_title(band_dict[band]["description"].title() + f" ({band})")
                ax.tick_params(labelsize=8)
                ax.spines.top.set_visible(False)
                ax.spines.right.set_visible(False)

            tabs = st.tabs(["Image", "Histogram"])
            with tabs[0]:
                st.pyplot(fig, use_container_width=True)

            with tabs[1]:
                st.pyplot(fig_hist, use_container_width=True)

    st.session_state.data = data

    return n_matches


def sentinel_2_band_process():
    data = st.session_state.data

    date_col, threshold_col = st.columns(2)
    real_color_col, fake_color_col, lw_color_col = st.columns(3)
    ndwi_col, mask_col = st.columns(2)

    ## User inputs
    with date_col:
        timestamp = st.selectbox(
            "Date Available",
            options=data.time.data,
            format_func=lambda x: str(x).partition("T")[0],
        )

    with threshold_col:
        threshold = st.number_input(
            "Water classification threshold",
            value=0.050,
            min_value=-0.5,
            max_value=0.5,
            step=0.01,
            format="%.3f",
            help="""
                Pixels with an NDWI less than the threshold given in 
                this field are classified as water.
            """,
        )

    ## Generate plots
    rgb, fake, land_water, ndwi = band_calculations(data.sel(time=timestamp))

    def wrap_imshow(data: np.ndarray, **imshow_kwargs):
        fig, ax = plt.subplots()
        ax.imshow(data, **imshow_kwargs)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return fig

    with real_color_col:
        st.markdown("**Real color**", help="Combines red, green and blue bands.")

        with st.spinner("Drawing image..."):
            img = wrap_imshow(rgb)

        st.pyplot(img, use_container_width=True)

    with fake_color_col:
        st.markdown(
            "**Fake color**",
            help="Combines near-infrared, red and green bands.",
        )

        with st.spinner("Drawing image..."):
            img = wrap_imshow(fake)

        st.pyplot(img, use_container_width=True)

    with lw_color_col:
        st.markdown(
            "**Land-Water**",
            help="Combines near-infrared, short-wave infrared and red bands.",
        )
        st.pyplot(wrap_imshow(land_water), use_container_width=True)

    with ndwi_col:
        st.markdown(
            "**NDWI**",
            help="Normalized Difference Water Index\n\n"
            "$$\n"
            R"\frac{\textsf{green} - \textsf{nir}}{\textsf{green} + \textsf{nir}}"
            "\n$$",
        )

        st.pyplot(wrap_imshow(ndwi, vmin=-0.25, vmax=0.25), use_container_width=True)

    with mask_col:
        st.markdown(
            "**Masked**", help="Classifies NDWI based on the threshold provided"
        )
        cmap = mpl.colors.ListedColormap(["w"]).with_extremes(
            under="#ff19ff10", over="#006994"
        )

        st.pyplot(
            wrap_imshow(
                ndwi, vmin=threshold - 0.001, vmax=threshold + 0.001, cmap=cmap
            ),
            use_container_width=True,
        )

    # Water coverage area
    water_pix = np.sum(np.where(ndwi > threshold, 1, 0))
    total_pix = ndwi.shape[0] * ndwi.shape[1]
    percent_pix = water_pix / total_pix
    area_water = total_pix * (10 * 10) / 1_000_000

    st.info(
        f"- :blue-background[**{total_pix}**] pixels were classified as water.\n"
        f"- They represent :blue-background[**{percent_pix:.0%}**] of the image.\n"
        f"- They account for :blue-background[**{area_water:.2f} km²**]"
    )

    # im = Image.fromarray((rgb * 255).astype('uint8'), mode="RGB")
    # st.image(im)


def sentinel_1_query():
    collection_details = st.session_state.catalog.get_collection(
        st.session_state.collection
    )

    query = st.session_state.catalog.search(
        collections=[st.session_state.collection],
        bbox=st.session_state.bbox,
        datetime=st.session_state.date_range,
    )

    items = list(query.items())
    n_matches = len(items)

    st.markdown(f":green-background[**{n_matches}** items were found.]")

    if n_matches:
        with st.spinner("Loading data..."):
            data = stac_load(
                items,
                bbox=st.session_state.bbox,
                groupby="solar_day",
                chunks={},  # <-- use Dask
            )

        tools_cols = st.columns([3, 1])
        image_placeholder = st.empty()

        with st.expander(
            f"See the **`{st.session_state.collection}`** collection description"
        ):
            st.html(collection_details._repr_html_())

        with st.expander("See the **`xarray`** dataset"):
            st.html(data._repr_html_())

        with tools_cols[0]:
            band = st.selectbox(
                "Bands Available", options=["vh", "vv"], format_func=str.upper
            )

        with tools_cols[1]:
            timestamp = st.selectbox(
                "Date Available",
                options=data.time.data,
                format_func=lambda x: str(x).partition("T")[0],
            )

        with image_placeholder:
            with st.spinner("Drawing images..."):
                fig, ax = plt.subplots(figsize=[6, 4.8])
                (np.log10(data[band].sel(time=timestamp)) * 10).plot.imshow(
                    ax=ax, cmap="Spectral_r", cbar_kwargs=dict(shrink=0.5)
                )
                ax.set_aspect("equal")
                ax.set_title(
                    datetime_as_string(timestamp, unit="m").replace("T", " | ")
                )
                ax.xaxis.label.set_visible(False)
                ax.yaxis.label.set_visible(False)
                ax.tick_params(labelsize=4)
                ax.tick_params(axis="x", labelrotation=90)
                ax.ticklabel_format(useOffset=False, style="plain")

                fig_hist, ax = plt.subplots(figsize=[6, 4])
                ax.hist(
                    np.log10(data[band].sel(time=timestamp).to_numpy().flatten()),
                    bins="scott",
                    color="#43a2ca",
                )
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.set_title(band.upper())
                ax.tick_params(labelsize=8)
                ax.spines.top.set_visible(False)
                ax.spines.right.set_visible(False)

            tabs = st.tabs(["Image", "Histogram"])
            with tabs[0]:
                st.pyplot(fig, use_container_width=True)

            with tabs[1]:
                st.pyplot(fig_hist, use_container_width=True)

        st.session_state.data = data
    return n_matches


def sentinel_1_band_process():
    data = st.session_state.data

    date_col, threshold_col = st.columns(2)
    vv_band_col, vh_band_col, rat_band_col = st.columns(3)
    rgb_band_col, threshold_col = st.columns(2)

    ## User inputs
    with date_col:
        timestamp = st.selectbox(
            "Date Available",
            options=data.time.data,
            format_func=lambda x: str(x).partition("T")[0],
        )

    ## Generate plots
    vv = data["vv"].sel(time=timestamp)
    vh = data["vh"].sel(time=timestamp)
    ratio = vh / vv

    # To dB
    log_vv = np.log10(vv) * 10
    log_vh = np.log10(vh) * 10

    def wrap_imshow(data: np.ndarray, **imshow_kwargs):
        fig, ax = plt.subplots()
        ax.imshow(data, **imshow_kwargs)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return fig

    with vv_band_col:
        st.markdown("**VV**", help="VV: vertical transmit, vertical receive")

        with st.spinner("Drawing image..."):
            img = wrap_imshow(log_vv, cmap="Greys_r")

        st.pyplot(img, use_container_width=True)

    with vh_band_col:
        st.markdown(
            "**VH**",
            help="VH: vertical transmit, horizontal receive",
        )

        with st.spinner("Drawing image..."):
            img = wrap_imshow(log_vh, cmap="Greys_r")

        st.pyplot(img, use_container_width=True)

    with rat_band_col:
        st.markdown(
            "**VH/VV**",
            help="Combines VH and VV bands",
        )
        with st.spinner("Drawing image..."):
            img = wrap_imshow(ratio, cmap="Greys_r")

        st.pyplot(img, use_container_width=True)

    with rgb_band_col:
        st.markdown(
            "**RGB**",
            help="Combines VH, VV and VH/VV bands",
        )
        with st.spinner("Drawing image..."):
            img = wrap_imshow(rgb_from_radar(data.sel(time=timestamp)), cmap="Greys_r")

        st.pyplot(img, use_container_width=True)

    with threshold_col:
        st.markdown(
            "**Threshold**",
            help="Pixels with an VV value less than the threshold given in this field are classified as water.",
        )

        with st.spinner("Drawing image..."):
            cmap = mpl.colors.ListedColormap(["w"]).with_extremes(
                over="#ff19ff10", under="#006994"
            )

            threshold = threshold_minimum(log_vv.to_numpy().flatten())
            img = wrap_imshow(
                log_vv, vmin=threshold - 0.001, vmax=threshold + 0.001, cmap=cmap
            )
        st.pyplot(img, use_container_width=True)

    # Water coverage area
    water_pix = np.sum(np.where(log_vv < threshold, 1, 0))
    total_pix = log_vv.shape[0] * log_vv.shape[1]
    percent_pix = water_pix / total_pix
    area_water = total_pix * (10 * 10) / 1_000_000

    st.info(
        f"- :blue-background[**{total_pix}**] pixels were classified as water.\n"
        f"- They represent :blue-background[**{percent_pix:.0%}**] of the image.\n"
        f"- They account for :blue-background[**{area_water:.2f} km²**]"
    )
