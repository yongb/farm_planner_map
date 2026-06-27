"""
Milk Hauling Route Planner — Streamlit Web App
---------------------------------------
- Visualizes all farms (cow icons), plants (factory icons), and haulers (truck icons)
  from farms.csv, plants.csv, haulers.csv on an interactive map.
- Each hauler is rendered in its own distinct color.
- Users pick farms / plants / haulers from filterable lists.
- Selected sites form an ordered itinerary (reorder, swap, remove).
- "Build Route" calls OpenRouteService (driving-hgv) to draw the real road route.

Run locally:
    pip install streamlit pandas folium streamlit-folium openrouteservice
    streamlit run sites_route_app.py
"""

import os
import colorsys
import pandas as pd
import streamlit as st
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
import openrouteservice as ors

# Icons (as Unicode escapes so the file is ASCII-safe)
COW = "\U0001F404"
FACTORY = "\U0001F3ED"
TRUCK = "\U0001F69A"
TRACTOR = "\U0001F69C"
ROAD = "\U0001F6E3"
BROOM = "\U0001F9F9"
SWAP_ICON = "\U0001F501"
PLUS = "\u2795"
CHECK = "\u2705"
INFO = "\u2139"
UP = "\u2B06"
DOWN = "\u2B07"
CROSS = "\u2716"

ORS_API_KEY = os.getenv(
    "ORS_API_KEY",
    "5b3ce3597851110001cf624892250d505d8f46feab89906e2c3e7b22",
)
FARMS_PATH = "farms.csv"
PLANTS_PATH = "plants.csv"
HAULERS_PATH = "haulers.csv"
ORS_MAX_WAYPOINTS = 50

st.set_page_config(page_title="Milk Hauling Route Planner", page_icon=TRACTOR, layout="wide")


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_farms(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "farm_lat": "lat", "farm_lon": "lng",
        "farm_id": "id", "farm_name": "name",
        "SourceName": "source_name",
        "farm_region": "region", "farm_type": "farm_type",
        "two_day_pounds": "two_day_pounds", "optilogic_county": "county",
    })
    for c in ["lat", "lng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    df["uid"] = "F-" + df["id"].astype(str) + "-" + df.index.astype(str)
    df["display"] = (
        df["source_name"].fillna(df["name"]).astype(str)
        + "  (" + df["id"].astype(str) + ")"
    )
    df["kind"] = "Farm"
    return df


@st.cache_data(show_spinner=False)
def load_plants(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "plant_lat": "lat", "plant_long": "lng",
        "plant_id": "id", "DestinationName": "name",
        "farm_region": "region",
    })
    for c in ["lat", "lng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    df["uid"] = "P-" + df["id"].astype(str)
    df["display"] = df["name"].astype(str) + "  (" + df["id"].astype(str) + ")"
    df["kind"] = "Plant"
    return df


@st.cache_data(show_spinner=False)
def load_haulers(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Latitude - Searched by Address": "lat",
        "Longitude - Searched by Address": "lng",
        "Hauler Name": "name", "Hauler ID": "id",
        "Region": "region", "City ": "city", "State": "state",
    })
    for c in ["lat", "lng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    df["uid"] = "H-" + df["id"].astype(str)
    df["display"] = df["name"].astype(str) + "  (" + df["id"].astype(str) + ")"
    df["kind"] = "Hauler"
    return df


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def distinct_color(i):
    h = (i * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.85)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


@st.cache_resource(show_spinner=False)
def get_ors_client(api_key):
    return ors.Client(key=api_key)


def get_route_coords(client, coords):
    """User-provided helper. coords is list of (lng, lat) tuples."""
    try:
        route = client.directions(coords, profile="driving-hgv", format="geojson")
        return (
            route["features"][0]["geometry"]["coordinates"],
            route["features"][0]["properties"]["summary"]["distance"],
        )
    except Exception:
        return [], 0


def build_route_chunked(client, coords):
    if len(coords) < 2:
        return [], 0, []
    all_coords, total_distance, errors = [], 0.0, []
    step = ORS_MAX_WAYPOINTS - 1
    for start in range(0, len(coords) - 1, step):
        end = min(start + ORS_MAX_WAYPOINTS, len(coords))
        chunk = coords[start:end]
        if len(chunk) < 2:
            break
        seg, dist = get_route_coords(client, chunk)
        if not seg:
            errors.append("segment " + str(start) + "-" + str(end))
            continue
        if all_coords:
            seg = seg[1:]
        all_coords.extend(seg)
        total_distance += dist
        if end == len(coords):
            break
    return all_coords, total_distance, errors


def emoji_icon_html(emoji, size=26):
    return (
        '<div style="font-size:' + str(size) + 'px;line-height:' + str(size)
        + 'px;text-shadow:0 1px 2px rgba(0,0,0,.35);">' + emoji + '</div>'
    )


def truck_icon_html(color, size=24):
    outer = size + 10
    inner = size - 4
    return (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'background:' + color + ';border:2px solid white;border-radius:50%;'
        'width:' + str(outer) + 'px;height:' + str(outer) + 'px;'
        'box-shadow:0 1px 6px rgba(0,0,0,.35);">'
        '<div style="font-size:' + str(inner) + 'px;line-height:' + str(inner) + 'px;">'
        + TRUCK + '</div></div>'
    )


def selected_badge_html(idx):
    return (
        '<div style="background:#16a34a;color:white;border-radius:50%;'
        'width:24px;height:24px;display:flex;align-items:center;'
        'justify-content:center;font-weight:700;border:2px solid white;'
        'box-shadow:0 1px 4px rgba(0,0,0,.4);">' + str(idx) + '</div>'
    )


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
farms = load_farms(FARMS_PATH)
plants = load_plants(PLANTS_PATH)
haulers = load_haulers(HAULERS_PATH)

# Per-hauler color map
hauler_ids_sorted = sorted(haulers["id"].astype(str).unique())
color_map = {hid: distinct_color(i) for i, hid in enumerate(hauler_ids_sorted)}
haulers["color"] = haulers["id"].astype(str).map(color_map)

all_sites = pd.concat([
    farms[["uid", "kind", "display", "lat", "lng", "region"]],
    plants[["uid", "kind", "display", "lat", "lng", "region"]],
    haulers[["uid", "kind", "display", "lat", "lng", "region"]],
], ignore_index=True)
uid_lookup = {row["uid"]: row.to_dict() for _, row in all_sites.iterrows()}


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []
if "route_geo" not in st.session_state:
    st.session_state.route_geo = None
    st.session_state.route_distance_m = 0
    st.session_state.route_errors = []


def add_to_itinerary(uids):
    for u in uids:
        if u not in st.session_state.itinerary:
            st.session_state.itinerary.append(u)


def remove_from_itinerary(uid):
    if uid in st.session_state.itinerary:
        st.session_state.itinerary.remove(uid)
    st.session_state.route_geo = None


def move(uid, delta):
    lst = st.session_state.itinerary
    i = lst.index(uid)
    j = max(0, min(len(lst) - 1, i + delta))
    if i != j:
        lst[i], lst[j] = lst[j], lst[i]
        st.session_state.route_geo = None


def swap(i, j):
    lst = st.session_state.itinerary
    if 0 <= i < len(lst) and 0 <= j < len(lst):
        lst[i], lst[j] = lst[j], lst[i]
        st.session_state.route_geo = None


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title(TRACTOR + " Milk Hauling Route Planner")
st.sidebar.caption("Pick farms, plants, and haulers; reorder the itinerary; build a real road route.")

with st.sidebar.expander(COW + " Farms (" + format(len(farms), ",") + ")", expanded=False):
    region_f = st.multiselect(
        "Farm region", sorted(farms["region"].dropna().unique()),
        default=sorted(farms["region"].dropna().unique()), key="region_f",
    )
    search_f = st.text_input("Search farms", key="search_f", placeholder="Name or ID...")
    f_filtered = farms[farms["region"].isin(region_f)]
    if search_f:
        s = search_f.lower()
        f_filtered = f_filtered[f_filtered["display"].str.lower().str.contains(s, na=False)]
    f_opts = f_filtered.sort_values("display")["display"].tolist()
    f_pick = st.multiselect("Select farms (" + str(len(f_opts)) + " shown)", options=f_opts, key="pick_f")
    if st.button(PLUS + " Add selected farms", use_container_width=True, key="add_f"):
        f_lookup = dict(zip(f_filtered["display"], f_filtered["uid"]))
        add_to_itinerary([f_lookup[d] for d in f_pick if d in f_lookup])

with st.sidebar.expander(FACTORY + " Plants (" + format(len(plants), ",") + ")", expanded=False):
    region_p = st.multiselect(
        "Plant region", sorted(plants["region"].dropna().unique()),
        default=sorted(plants["region"].dropna().unique()), key="region_p",
    )
    search_p = st.text_input("Search plants", key="search_p", placeholder="Name or ID...")
    p_filtered = plants[plants["region"].isin(region_p)]
    if search_p:
        s = search_p.lower()
        p_filtered = p_filtered[p_filtered["display"].str.lower().str.contains(s, na=False)]
    p_opts = p_filtered.sort_values("display")["display"].tolist()
    p_pick = st.multiselect("Select plants (" + str(len(p_opts)) + " shown)", options=p_opts, key="pick_p")
    if st.button(PLUS + " Add selected plants", use_container_width=True, key="add_p"):
        p_lookup = dict(zip(p_filtered["display"], p_filtered["uid"]))
        add_to_itinerary([p_lookup[d] for d in p_pick if d in p_lookup])

with st.sidebar.expander(TRUCK + " Haulers (" + format(len(haulers), ",") + ")", expanded=True):
    region_h = st.multiselect(
        "Hauler region", sorted(haulers["region"].dropna().unique()),
        default=sorted(haulers["region"].dropna().unique()), key="region_h",
    )
    search_h = st.text_input("Search haulers", key="search_h", placeholder="Name, ID, city, state...")
    h_filtered = haulers[haulers["region"].isin(region_h)]
    if search_h:
        s = search_h.lower()
        h_filtered = h_filtered[
            h_filtered["display"].str.lower().str.contains(s, na=False)
            | h_filtered["city"].astype(str).str.lower().str.contains(s, na=False)
            | h_filtered["state"].astype(str).str.lower().str.contains(s, na=False)
        ]
    h_opts = h_filtered.sort_values("display")["display"].tolist()
    h_pick = st.multiselect("Select haulers (" + str(len(h_opts)) + " shown)", options=h_opts, key="pick_h")
    if st.button(PLUS + " Add selected haulers", use_container_width=True, key="add_h"):
        h_lookup = dict(zip(h_filtered["display"], h_filtered["uid"]))
        add_to_itinerary([h_lookup[d] for d in h_pick if d in h_lookup])

st.sidebar.markdown("---")
col_r1, col_r2 = st.sidebar.columns(2)
build_clicked = col_r1.button(ROAD + " Build Route", type="primary", use_container_width=True)
if col_r2.button(BROOM + " Clear itinerary", use_container_width=True):
    st.session_state.itinerary = []
    st.session_state.route_geo = None


# ------------------------------------------------------------
# Header / KPIs
# ------------------------------------------------------------
st.title("Milk Hauling Route Planner")
k1, k2, k3, k4 = st.columns(4)
k1.metric(COW + " Farms", format(len(farms), ","))
k2.metric(FACTORY + " Plants", format(len(plants), ","))
k3.metric(TRUCK + " Haulers", format(len(haulers), ","))
k4.metric("Selected stops", len(st.session_state.itinerary))


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
left, right = st.columns([3, 2])

with right:
    st.subheader("Itinerary (route order)")
    itin = st.session_state.itinerary
    if not itin:
        st.info("Use the sidebar to add farms, plants, or haulers to your itinerary.")
    else:
        sw1, sw2, sw3 = st.columns([1, 1, 1])
        with sw1:
            i_swap = st.number_input("Swap pos", min_value=1, max_value=len(itin), value=1, key="sw_i")
        with sw2:
            j_swap = st.number_input(
                "with pos", min_value=1, max_value=len(itin),
                value=min(2, len(itin)), key="sw_j",
            )
        with sw3:
            st.write("")
            if st.button(SWAP_ICON + " Swap", use_container_width=True):
                swap(int(i_swap) - 1, int(j_swap) - 1)
                st.rerun()

        badges = {"Farm": COW, "Plant": FACTORY, "Hauler": TRUCK}
        for idx, uid in enumerate(list(itin), start=1):
            site = uid_lookup.get(uid, {})
            kind = site.get("kind", "?")
            badge = badges.get(kind, "*")
            c1, c2, c3, c4, c5 = st.columns([0.55, 4, 0.6, 0.6, 0.6])
            c1.markdown("**" + str(idx) + ".**")
            label = (
                badge + " **" + str(site.get("display", uid)) + "**  \n"
                "<small>" + str(kind) + " - " + str(site.get("region", "")) + "</small>"
            )
            c2.markdown(label, unsafe_allow_html=True)
            if c3.button(UP, key="up_" + uid, help="Move up"):
                move(uid, -1)
                st.rerun()
            if c4.button(DOWN, key="dn_" + uid, help="Move down"):
                move(uid, +1)
                st.rerun()
            if c5.button(CROSS, key="rm_" + uid, help="Remove"):
                remove_from_itinerary(uid)
                st.rerun()

    if st.session_state.route_geo:
        miles = st.session_state.route_distance_m / 1609.344
        km = st.session_state.route_distance_m / 1000.0
        st.success("Road route: **{:,.1f} mi** ({:,.1f} km)".format(miles, km))
        if st.session_state.route_errors:
            st.warning("Some route segments failed: " + ", ".join(st.session_state.route_errors))


if build_clicked:
    stops = [uid_lookup[u] for u in st.session_state.itinerary if u in uid_lookup]
    if len(stops) < 2:
        st.warning("Add at least two sites to the itinerary before building a route.")
    else:
        coords = [(s["lng"], s["lat"]) for s in stops]
        with st.spinner("Requesting road route for " + str(len(stops)) + " stops..."):
            client = get_ors_client(ORS_API_KEY)
            geo, dist_m, errs = build_route_chunked(client, coords)
        if not geo:
            st.error(
                "Could not retrieve a road route from OpenRouteService. "
                "Check the daily quota or try fewer stops."
            )
        else:
            st.session_state.route_geo = geo
            st.session_state.route_distance_m = dist_m
            st.session_state.route_errors = errs


with left:
    st.subheader("Map")

    center_lat = float(pd.concat([farms["lat"], plants["lat"], haulers["lat"]]).mean())
    center_lng = float(pd.concat([farms["lng"], plants["lng"], haulers["lng"]]).mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=5, tiles="cartodbpositron")

    # Farms
    farms_layer = folium.FeatureGroup(name=COW + " Farms", show=True)
    for _, r in farms.iterrows():
        popup_html = (
            "<b>" + COW + " " + str(r["display"]) + "</b><br>"
            "Region: " + str(r.get("region", "")) + "<br>"
            "County: " + str(r.get("county", "")) + "<br>"
            "Type: " + str(r.get("farm_type", ""))
        )
        folium.Marker(
            [r["lat"], r["lng"]],
            tooltip=COW + " " + str(r["display"]),
            popup=folium.Popup(popup_html, max_width=320),
            icon=DivIcon(icon_size=(28, 28), icon_anchor=(14, 14),
                         html=emoji_icon_html(COW)),
        ).add_to(farms_layer)
    farms_layer.add_to(m)

    # Plants
    plants_layer = folium.FeatureGroup(name=FACTORY + " Plants", show=True)
    for _, r in plants.iterrows():
        popup_html = (
            "<b>" + FACTORY + " " + str(r["display"]) + "</b><br>"
            "Region: " + str(r.get("region", ""))
        )
        folium.Marker(
            [r["lat"], r["lng"]],
            tooltip=FACTORY + " " + str(r["display"]),
            popup=folium.Popup(popup_html, max_width=320),
            icon=DivIcon(icon_size=(28, 28), icon_anchor=(14, 14),
                         html=emoji_icon_html(FACTORY)),
        ).add_to(plants_layer)
    plants_layer.add_to(m)

    # Haulers
    haulers_layer = folium.FeatureGroup(name=TRUCK + " Haulers", show=True)
    for _, r in haulers.iterrows():
        popup_html = (
            "<b>" + TRUCK + " " + str(r["display"]) + "</b><br>"
            "Region: " + str(r.get("region", "")) + "<br>"
            + str(r.get("city", "")) + ", " + str(r.get("state", ""))
        )
        folium.Marker(
            [r["lat"], r["lng"]],
            tooltip=TRUCK + " " + str(r["display"]),
            popup=folium.Popup(popup_html, max_width=320),
            icon=DivIcon(icon_size=(34, 34), icon_anchor=(17, 17),
                         html=truck_icon_html(r["color"])),
        ).add_to(haulers_layer)
    haulers_layer.add_to(m)

    # Selected stops
    if st.session_state.itinerary:
        sel_layer = folium.FeatureGroup(name=CHECK + " Selected stops", show=True)
        for idx, uid in enumerate(st.session_state.itinerary, start=1):
            s = uid_lookup.get(uid)
            if not s:
                continue
            folium.CircleMarker(
                [s["lat"], s["lng"]],
                radius=18, color="#16a34a", weight=3, fill=False,
            ).add_to(sel_layer)
            folium.Marker(
                [s["lat"], s["lng"]],
                icon=DivIcon(icon_size=(28, 28), icon_anchor=(14, -10),
                             html=selected_badge_html(idx)),
            ).add_to(sel_layer)
        sel_layer.add_to(m)

    # Route polyline
    if st.session_state.route_geo:
        line = [[c[1], c[0]] for c in st.session_state.route_geo]
        folium.PolyLine(line, color="#15803d", weight=6, opacity=0.85,
                        tooltip="Road route (driving-hgv)").add_to(m)
        lats = [p[0] for p in line]
        lngs = [p[1] for p in line]
        m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    elif st.session_state.itinerary:
        lats = [uid_lookup[u]["lat"] for u in st.session_state.itinerary if u in uid_lookup]
        lngs = [uid_lookup[u]["lng"] for u in st.session_state.itinerary if u in uid_lookup]
        if lats:
            m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, use_container_width=True, height=720, returned_objects=[])


with st.expander(INFO + " Tips"):
    st.markdown(
        "- **Icons**: " + COW + " farms, " + FACTORY + " plants, " + TRUCK
        + " haulers (each hauler has a distinct color).\n"
        "- **Selecting**: open a section in the sidebar, pick rows in the multi-select, then click *Add*.\n"
        "- **Reordering**: use " + UP + "/" + DOWN
        + " next to each itinerary row, or the **Swap pos / with pos** controls.\n"
        "- **Remove**: the " + CROSS + " next to each row removes that stop.\n"
        "- **Build Route**: real driving-hgv road route via OpenRouteService; the green polyline is the result.\n"
        "- **Long routes**: ORS allows up to 50 waypoints per request; the app automatically splits longer "
        "itineraries into overlapping segments and stitches them together."
    )
