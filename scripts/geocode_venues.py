"""One-time generator for config/venue_coords.yaml (P7 venue-coords mini-task).

Geocodes the ATP GS / Masters1000 / ATP500 / ATP250 tournament venues via
Nominatim (free, no API key, 1 req/sec policy) and writes a static, manually
reviewable YAML. This is a DEV utility, NOT a runtime dependency — DECISIONS.md
§L11 forbids runtime geocoding. The DataAgent reads the generated YAML and
upserts venues idempotently; it never calls a geocoder.

Always queries "City, Country" (both parts) to avoid misfires (e.g. Washington
DC vs Washington state, Lyon vs other Lyons).

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/geocode_venues.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

# Each venue: slug/name/city/country/country_code are authored; lat/lon are
# filled by geocoding. `query` overrides the default "city, country" lookup when
# the bare city name is ambiguous. indoor/surface are tournament metadata kept in
# the YAML for future tournament linkage (the venues table has neither column).
VENUES: list[dict[str, Any]] = [
    # --- Grand Slams (4) ---
    {"slug": "australian-open", "name": "Australian Open", "city": "Melbourne", "country": "Australia", "country_code": "AUS", "indoor": False, "surface": "Hard"},
    {"slug": "roland-garros", "name": "Roland Garros", "city": "Paris", "country": "France", "country_code": "FRA", "indoor": False, "surface": "Clay"},
    {"slug": "wimbledon", "name": "Wimbledon", "city": "London", "country": "United Kingdom", "country_code": "GBR", "indoor": False, "surface": "Grass"},
    {"slug": "us-open", "name": "US Open", "city": "New York", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Hard"},
    # --- Masters 1000 (9) ---
    {"slug": "indian-wells", "name": "Indian Wells Masters", "city": "Indian Wells", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Hard"},
    {"slug": "miami-open", "name": "Miami Open", "city": "Miami", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Hard"},
    {"slug": "monte-carlo", "name": "Monte-Carlo Masters", "city": "Monte-Carlo", "country": "Monaco", "country_code": "MCO", "indoor": False, "surface": "Clay"},
    {"slug": "madrid-open", "name": "Madrid Open", "city": "Madrid", "country": "Spain", "country_code": "ESP", "indoor": False, "surface": "Clay"},
    {"slug": "italian-open", "name": "Italian Open", "city": "Rome", "country": "Italy", "country_code": "ITA", "indoor": False, "surface": "Clay"},
    {"slug": "canadian-open", "name": "Canadian Open", "city": "Toronto", "country": "Canada", "country_code": "CAN", "indoor": False, "surface": "Hard"},
    {"slug": "cincinnati", "name": "Western & Southern Open", "city": "Cincinnati", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Hard"},
    {"slug": "shanghai-masters", "name": "Shanghai Masters", "city": "Shanghai", "country": "China", "country_code": "CHN", "indoor": False, "surface": "Hard"},
    {"slug": "paris-masters", "name": "Paris Masters", "city": "Paris", "country": "France", "country_code": "FRA", "indoor": True, "surface": "Hard"},
    # --- ATP 500 (16) ---
    {"slug": "dallas-open", "name": "Dallas Open", "city": "Frisco", "country": "United States", "country_code": "USA", "query": "Frisco, Texas, United States", "indoor": True, "surface": "Hard"},
    {"slug": "rotterdam-open", "name": "Rotterdam Open", "city": "Rotterdam", "country": "Netherlands", "country_code": "NLD", "indoor": True, "surface": "Hard"},
    {"slug": "qatar-open", "name": "Qatar Open", "city": "Doha", "country": "Qatar", "country_code": "QAT", "indoor": True, "surface": "Hard"},
    {"slug": "rio-open", "name": "Rio Open", "city": "Rio de Janeiro", "country": "Brazil", "country_code": "BRA", "indoor": False, "surface": "Clay"},
    {"slug": "dubai-championships", "name": "Dubai Championships", "city": "Dubai", "country": "United Arab Emirates", "country_code": "ARE", "indoor": True, "surface": "Hard"},
    {"slug": "mexican-open", "name": "Mexican Open", "city": "Acapulco", "country": "Mexico", "country_code": "MEX", "indoor": False, "surface": "Hard"},
    {"slug": "barcelona-open", "name": "Barcelona Open", "city": "Barcelona", "country": "Spain", "country_code": "ESP", "indoor": False, "surface": "Clay"},
    {"slug": "bavarian-international", "name": "Bavarian International", "city": "Munich", "country": "Germany", "country_code": "DEU", "indoor": False, "surface": "Clay"},
    {"slug": "hamburg-open", "name": "Hamburg Open", "city": "Hamburg", "country": "Germany", "country_code": "DEU", "indoor": False, "surface": "Clay"},
    {"slug": "queens-club", "name": "Queen's Club Championships", "city": "London", "country": "United Kingdom", "country_code": "GBR", "indoor": False, "surface": "Grass"},
    {"slug": "halle-open", "name": "Halle Open", "city": "Halle", "country": "Germany", "country_code": "DEU", "query": "Halle, Westphalia, Germany", "indoor": False, "surface": "Grass"},
    {"slug": "washington-open", "name": "Washington Open", "city": "Washington", "country": "United States", "country_code": "USA", "query": "Washington, D.C., United States", "indoor": False, "surface": "Hard"},
    {"slug": "japan-open", "name": "Japan Open", "city": "Tokyo", "country": "Japan", "country_code": "JPN", "indoor": False, "surface": "Hard"},
    {"slug": "china-open", "name": "China Open", "city": "Beijing", "country": "China", "country_code": "CHN", "indoor": False, "surface": "Hard"},
    {"slug": "vienna-open", "name": "Vienna Open", "city": "Vienna", "country": "Austria", "country_code": "AUT", "indoor": True, "surface": "Hard"},
    {"slug": "swiss-indoors", "name": "Swiss Indoors", "city": "Basel", "country": "Switzerland", "country_code": "CHE", "indoor": True, "surface": "Hard"},
    # --- ATP 250 (2026 calendar) ---
    {"slug": "auckland", "name": "Auckland", "city": "Auckland", "country": "New Zealand", "country_code": "NZL", "indoor": False, "surface": "Hard"},
    {"slug": "adelaide", "name": "Adelaide", "city": "Adelaide", "country": "Australia", "country_code": "AUS", "indoor": False, "surface": "Hard"},
    {"slug": "brisbane", "name": "Brisbane", "city": "Brisbane", "country": "Australia", "country_code": "AUS", "indoor": False, "surface": "Hard"},
    {"slug": "montpellier", "name": "Montpellier", "city": "Montpellier", "country": "France", "country_code": "FRA", "indoor": True, "surface": "Hard"},
    {"slug": "buenos-aires", "name": "Buenos Aires", "city": "Buenos Aires", "country": "Argentina", "country_code": "ARG", "indoor": False, "surface": "Clay"},
    {"slug": "santiago", "name": "Santiago", "city": "Santiago", "country": "Chile", "country_code": "CHL", "indoor": False, "surface": "Clay"},
    {"slug": "delray-beach", "name": "Delray Beach", "city": "Delray Beach", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Hard"},
    {"slug": "marrakech", "name": "Marrakech", "city": "Marrakech", "country": "Morocco", "country_code": "MAR", "indoor": False, "surface": "Clay"},
    {"slug": "estoril", "name": "Estoril", "city": "Estoril", "country": "Portugal", "country_code": "PRT", "indoor": False, "surface": "Clay"},
    {"slug": "geneva", "name": "Geneva", "city": "Geneva", "country": "Switzerland", "country_code": "CHE", "indoor": False, "surface": "Clay"},
    {"slug": "lyon", "name": "Lyon", "city": "Lyon", "country": "France", "country_code": "FRA", "indoor": False, "surface": "Clay"},
    {"slug": "stuttgart", "name": "Stuttgart", "city": "Stuttgart", "country": "Germany", "country_code": "DEU", "indoor": False, "surface": "Grass"},
    {"slug": "eastbourne", "name": "Eastbourne", "city": "Eastbourne", "country": "United Kingdom", "country_code": "GBR", "indoor": False, "surface": "Grass"},
    {"slug": "s-hertogenbosch", "name": "'s-Hertogenbosch", "city": "'s-Hertogenbosch", "country": "Netherlands", "country_code": "NLD", "indoor": False, "surface": "Grass"},
    {"slug": "mallorca", "name": "Mallorca", "city": "Mallorca", "country": "Spain", "country_code": "ESP", "query": "Palma, Mallorca, Spain", "indoor": False, "surface": "Grass"},
    {"slug": "bastad", "name": "Bastad", "city": "Bastad", "country": "Sweden", "country_code": "SWE", "query": "Bastad, Sweden", "indoor": False, "surface": "Clay"},
    {"slug": "gstaad", "name": "Gstaad", "city": "Gstaad", "country": "Switzerland", "country_code": "CHE", "indoor": False, "surface": "Clay"},
    {"slug": "umag", "name": "Umag", "city": "Umag", "country": "Croatia", "country_code": "HRV", "indoor": False, "surface": "Clay"},
    {"slug": "kitzbuhel", "name": "Kitzbuhel", "city": "Kitzbuhel", "country": "Austria", "country_code": "AUT", "query": "Kitzbuhel, Austria", "indoor": False, "surface": "Clay"},
    {"slug": "los-cabos", "name": "Los Cabos", "city": "Los Cabos", "country": "Mexico", "country_code": "MEX", "query": "San Jose del Cabo, Baja California Sur, Mexico", "indoor": False, "surface": "Hard"},
    {"slug": "winston-salem", "name": "Winston-Salem", "city": "Winston-Salem", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Hard"},
    {"slug": "chengdu", "name": "Chengdu", "city": "Chengdu", "country": "China", "country_code": "CHN", "indoor": False, "surface": "Hard"},
    {"slug": "almaty", "name": "Almaty", "city": "Almaty", "country": "Kazakhstan", "country_code": "KAZ", "indoor": False, "surface": "Hard"},
    {"slug": "hong-kong", "name": "Hong Kong", "city": "Hong Kong", "country": "Hong Kong", "country_code": "HKG", "indoor": False, "surface": "Hard"},
    {"slug": "bucharest", "name": "Bucharest", "city": "Bucharest", "country": "Romania", "country_code": "ROU", "indoor": False, "surface": "Clay"},
    {"slug": "houston", "name": "Houston", "city": "Houston", "country": "United States", "country_code": "USA", "indoor": False, "surface": "Clay"},
    {"slug": "hangzhou", "name": "Hangzhou", "city": "Hangzhou", "country": "China", "country_code": "CHN", "indoor": False, "surface": "Hard"},
    {"slug": "brussels", "name": "Brussels", "city": "Brussels", "country": "Belgium", "country_code": "BEL", "indoor": False, "surface": "Clay"},
    {"slug": "stockholm", "name": "Stockholm", "city": "Stockholm", "country": "Sweden", "country_code": "SWE", "indoor": True, "surface": "Hard"},
]

_FIELD_ORDER = (
    "slug", "name", "city", "country", "country_code",
    "lat", "lon", "altitude_m", "indoor", "surface",
)
_OUT = Path("config/venue_coords.yaml")


def main() -> None:
    geolocator = Nominatim(user_agent="tennis_prediction_bot_venue_geocoder", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    out: list[dict[str, Any]] = []
    misses: list[str] = []
    for v in VENUES:
        query = v.get("query") or f"{v['city']}, {v['country']}"
        loc = geocode(query)
        if loc is None:
            misses.append(f"{v['slug']} ({query})")
            lat = lon = None
        else:
            lat = round(loc.latitude, 4)
            lon = round(loc.longitude, 4)
        print(f"{v['slug']:<22} {query:<34} -> {lat}, {lon}")
        out.append({
            "slug": v["slug"], "name": v["name"], "city": v["city"],
            "country": v["country"], "country_code": v["country_code"],
            "lat": lat, "lon": lon, "altitude_m": None,
            "indoor": v["indoor"], "surface": v["surface"],
        })

    # Reorder keys deterministically for a clean, reviewable diff.
    ordered = [{k: e[k] for k in _FIELD_ORDER} for e in out]
    header = (
        "# Static venue coordinates for the OWM weather step (DECISIONS.md §L11).\n"
        "# Generated by scripts/geocode_venues.py (GeoPy/Nominatim, one-shot).\n"
        "# NOT consulted at runtime. DataAgent upserts these into `venues` (city-level,\n"
        "# keyed on city+country_code); indoor/surface are tournament metadata kept here\n"
        "# for future tournament linkage. Same-city events (Wimbledon+Queen's -> London;\n"
        "# Roland Garros+Paris Masters -> Paris) collapse to one venue on upsert.\n"
    )
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(ordered, fh, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"\nWrote {len(ordered)} entries to {_OUT}")
    if misses:
        print(f"!! {len(misses)} geocoding MISS(es) — fix manually: {misses}")


if __name__ == "__main__":
    main()
