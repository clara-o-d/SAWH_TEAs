import requests
import json
import time
import csv
import os

BASE_URL = "https://wabi-us-east2-api.analysis.windows.net/public/reports/querydata?synchronous=true"

# --- IDs pulled from your captured request ---
DATASET_ID = "74b2a186-2054-44e6-a714-4f9152abf878"
MODEL_ID = 7076798  # must be int

# NOTE: ReportId is almost certainly the same across visuals on one report page.
# VisualId is per-visual. If get_utilities() and get_category_value() correspond
# to two different charts/tables on the page, open devtools, trigger each one,
# and paste in its own VisualId below.
REPORT_ID = "240737dd-f0fc-451a-b410-193d3acbc2d0"
VISUAL_ID_UTILITIES = "367eb208cd0c407669aa"      # only used if APP_CONTEXT_MODE == "full"
VISUAL_ID_CATEGORY = "367eb208cd0c407669aa"       # only used if APP_CONTEXT_MODE == "full"

# Try these in order -- start with "none", move down only if you get a 400/403/anything != 200:
#   "none"       -> no ApplicationContext at all
#   "dataset"    -> ApplicationContext with DatasetId only, no Sources/VisualId
#   "full"       -> ApplicationContext with DatasetId + ReportId + VisualId (exact match to captured request)
APP_CONTEXT_MODE = "none"
DEBUG = False  # set True to dump raw JSON responses for troubleshooting

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    # REQUIRED: copy fresh from browser devtools -- these tokens expire (often ~1hr)
    "Authorization": "Bearer YOUR_TOKEN_HERE",
}
HEADERS["X-PowerBI-ResourceKey"] = "d3976ecf-dbda-4a9f-86a9-7657202e1d59"
COUNTRIES = [
    "Afghanistan",
"Albania",
"Algeria",
"American Samoa",
"Andorra",
"Angola",
"Anguilla",
"Antigua and Barbuda",
"Argentina",
"Armenia",
"Aruba",
"Australia",
"Austria",
"Azerbaijan",
"Bahamas",
"Bahrain",
"Bangladesh",
"Barbados",
"Belarus",
"Belgium",
"Belize",
"Benin",
"Bermuda",
"Bhutan",
"Bolivia",
"Bosnia and Herzegovina",
"Botswana",
"Brazil",
"British Virgin Islands",
"Brunei",
"Bulgaria",
"Burkina Faso",
"Burundi",
"Cambodia",
"Cameroon",
"Canada",
"Cape Verde",
"Cayman Islands",
"Central African Republic",
"Chad",
"Chile",
"China",
"Colombia",
"Comoros",
"Congo",
"Congo, Dem. Rep.",
"Cook Islands",
"Costa Rica",
"Cote d'Ivoire",
"Croatia",
"Cyprus",
"Czech Republic",
"Denmark",
"Djibouti",
"Dominica",
"Dominican Republic",
"Ecuador",
"Egypt",
"El Salvador",
"Eritrea",
"Estonia",
"Ethiopia",
"Federated States Of Micronesia",
"Fiji",
"Finland",
"France",
"French Polynesia",
"Gabon",
"Georgia",
"Germany",
"Ghana",
"Gibraltar",
"Greece",
"Greenland",
"Grenada",
"Guadeloupe",
"Guam",
"Guatemala",
"Guernsey",
"Guinea",
"Guinea-Bissau",
"Guyana",
"Haiti",
"Honduras",
"Hungary",
"Iceland",
"India",
"Indonesia",
"Iran",
"Iraq",
"Ireland",
"Isle of Man",
"Israel",
"Italy",
"Jamaica",
"Japan",
"Jersey",
"Jordan",
"Kazakhstan",
"Kenya",
"Kosovo",
"Kuwait",
"Kyrgyz Republic",
"Lao PDR",
"Latvia",
"Lebanon",
"Lesotho",
"Liberia",
"Libya",
"Liechtenstein",
"Lithuania",
"Luxembourg",
"Macau, China",
"Madagascar",
"Malawi",
"Malaysia",
"Maldives",
"Mali",
"Malta",
"Marshall Islands",
"Martinique",
"Mauritania",
"Mauritius",
"Mexico",
"Moldova",
"Monaco",
"Mongolia",
"Montenegro",
"Montserrat",
"Morocco",
"Mozambique",
"Myanmar",
"Namibia",
"Nepal",
"Netherlands",
"Netherlands Antilles",
"New Zealand",
"Niger",
"Nigeria",
"North Macedonia",
"Norway",
"Oman",
"Pakistan",
"Palau",
"Panama",
"Papua New Guinea",
"Paraguay",
"Peru",
"Philippines",
"Poland",
"Portugal",
"Puerto Rico",
"Qatar",
"Republic Of Kiribati",
"Republic Of Nauru",
"Reunion",
"Romania",
"Russia",
"Rwanda",
"Saint Helena",
"Saint Pierre and Miquelon",
"Samoa",
"San Marino",
"Saudi Arabia",
"Senegal",
"Serbia",
"Seychelles",
"Sierra Leone",
"Singapore",
"Slovakia",
"Slovenia",
"Solomon Islands",
"Somaliland",
"South Africa",
"South Korea",
"Spain",
"Sri Lanka",
"St. Kitts and Nevis",
"St. Vincent and the Grenadines",
"Sudan",
"Suriname",
"Swaziland",
"Sweden",
"Switzerland",
"Taiwan",
"Tajikistan",
"Tanzania",
"Thailand",
"The Gambia",
"Timor-Leste",
"Togo",
"Tonga",
"Trinidad and Tobago",
"Tunisia",
"Turkey",
"Turkmenistan",
"Turks and Caicos Islands",
"Tuvalu",
"U.S. Virgin Islands",
"Uganda",
"UK, England and Wales",
"UK, Scotland",
"Ukraine",
"United Arab Emirates",
"United States",
"Uruguay",
"Uzbekistan",
"Vanuatu",
"Venezuela",
"Vietnam",
"Wallis and Futuna",
"West Bank and Gaza",
"Zambia",
"Zimbabwe"
]
CATEGORIES = ["6M3", "15M3", "50M3"]


def post(payload):
    r = requests.post(BASE_URL, headers=HEADERS, json=payload)

    if r.status_code != 200:
        print("STATUS:", r.status_code)
        print(r.text)
        r.raise_for_status()

    return r.json()


def application_context(visual_id):
    """Returns None (omit key), a dataset-only dict, or the full dict,
    depending on APP_CONTEXT_MODE. See comment above APP_CONTEXT_MODE."""
    if APP_CONTEXT_MODE == "none":
        return None
    if APP_CONTEXT_MODE == "dataset":
        return {"DatasetId": DATASET_ID}
    if APP_CONTEXT_MODE == "full":
        return {
            "DatasetId": DATASET_ID,
            "Sources": [{"ReportId": REPORT_ID, "VisualId": visual_id}],
        }
    raise ValueError(f"Unknown APP_CONTEXT_MODE: {APP_CONTEXT_MODE}")


def build_query_wrapper(command, visual_id):
    """Wraps a Command dict into the outer 'queries' list entry,
    including ApplicationContext only if the current mode calls for it."""
    entry = {"Query": {"Commands": [command]}, "QueryId": ""}
    ctx = application_context(visual_id)
    if ctx is not None:
        entry["ApplicationContext"] = ctx
    return entry


# -----------------------------
# STEP 1: GET UTILITIES
# -----------------------------
def get_utilities(country):
    command = {
        "SemanticQueryDataShapeCommand": {
            "Query": {
                "Version": 2,
                "From": [
                    {"Name": "t", "Entity": "Test", "Type": 0}
                ],
                "Select": [
                    {
                        "Column": {
                            "Expression": {"SourceRef": {"Source": "t"}},
                            "Property": "Utility"
                        },
                        "Name": "Utility"
                    }
                ],
                "Where": [{
                    "Condition": {
                        "In": {
                            "Expressions": [{
                                "Column": {
                                    "Expression": {"SourceRef": {"Source": "t"}},
                                    "Property": "Country"
                                }
                            }],
                            "Values": [[{"Literal": {"Value": f"'{country}'"}}]]
                        }
                    }
                }]
            },
            "Binding": {
                "Primary": {"Groupings": [{"Projections": [0]}]},
                "DataReduction": {
                    "DataVolume": 3,
                    "Primary": {"Window": {"Count": 1000}}
                },
                "Version": 1
            }
        }
    }

    payload = {
        "version": "1.0.0",
        "queries": [build_query_wrapper(command, VISUAL_ID_UTILITIES)],
        "cancelQueries": [],
        "modelId": MODEL_ID,
    }

    res = post(payload)

    if DEBUG:
        print(json.dumps(res, indent=2)[:3000])

    ds0 = res["results"][0]["result"]["data"]["dsr"]["DS"][0]
    utilities = ds0.get("PH", [{}])[0].get("DM0", [])

    return sorted({u["G0"] for u in utilities if "G0" in u})


# -----------------------------
# STEP 2: GET CATEGORY VALUE
# -----------------------------
def get_category_value(country, utility, category):
    command = {
        "SemanticQueryDataShapeCommand": {
            "Query": {
                "Version": 2,
                "From": [
                    {"Name": "t", "Entity": "Test", "Type": 0},
                    {"Name": "m", "Entity": "Measure", "Type": 0}
                ],
                "Select": [
                    {
                        "Column": {
                            "Expression": {"SourceRef": {"Source": "t"}},
                            "Property": "Value"
                        },
                        "Name": "Value"
                    }
                ],
                "Where": [
                    {
                        "Condition": {
                            "In": {
                                "Expressions": [{
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "t"}},
                                        "Property": "Country"
                                    }
                                }],
                                "Values": [[{"Literal": {"Value": f"'{country}'"}}]]
                            }
                        }
                    },
                    {
                        "Condition": {
                            "In": {
                                "Expressions": [{
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "t"}},
                                        "Property": "Utility"
                                    }
                                }],
                                "Values": [[{"Literal": {"Value": f"'{utility}'"}}]]
                            }
                        }
                    },
                    {
                        "Condition": {
                            "In": {
                                "Expressions": [{
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "t"}},
                                        "Property": "Category"
                                    }
                                }],
                                "Values": [[{"Literal": {"Value": f"'{category}'"}}]]
                            }
                        }
                    }
                ]
            },
            "Binding": {
                "Primary": {"Groupings": [{"Projections": [0]}]},
                "DataReduction": {
                    "DataVolume": 3,
                    "Primary": {"Window": {"Count": 1000}}
                },
                "Version": 1
            }
        }
    }

    payload = {
        "version": "1.0.0",
        "queries": [build_query_wrapper(command, VISUAL_ID_CATEGORY)],
        "cancelQueries": [],
        "modelId": MODEL_ID,
    }

    res = post(payload)

    if DEBUG:
        print(json.dumps(res, indent=2)[:2000])

    ds0 = res["results"][0]["result"]["data"]["dsr"]["DS"][0]
    rows = ds0.get("PH", [])
    if not rows or "DM0" not in rows[0] or not rows[0]["DM0"]:
        return None

    row0 = rows[0]["DM0"][0]
    # Same-shape query as get_utilities, so PBI is expected to key the value
    # under "G0" rather than "C" -- keeping both as a fallback just in case.
    if "G0" in row0:
        return row0["G0"]
    if "C" in row0:
        return row0["C"][0]
    return None


# -----------------------------
# STEP 3: PIPELINE (writes to CSV as it goes, resumable)
# -----------------------------
def prepare_resume(filepath, countries):
    """Looks at the existing CSV (if any) and figures out where to resume.

    Strategy: find the LAST country that has any rows in the file (this may
    be a country that only got partially processed before a crash). Strip
    that country's rows out of the file entirely, then resume processing
    from that country onward -- so it gets fully redone rather than left
    half-finished, and everything before it is left untouched.

    Returns the list of countries still left to process.
    """
    try:
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except FileNotFoundError:
        return countries  # no existing file -> process everything, fresh file created later

    if not rows:
        return countries  # file exists but empty/header-only -> process everything

    last_country = rows[-1][0]
    print(f"Found existing {filepath}. Last country present: {last_country!r} -- will redo it fully.")

    # Drop every row belonging to that last (possibly-partial) country.
    kept_rows = [r for r in rows if r and r[0] != last_country]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header if header else ["Country", "Utility"] + CATEGORIES)
        writer.writerows(kept_rows)

    if last_country in countries:
        resume_index = countries.index(last_country)
    else:
        # Last country in the CSV isn't in our current COUNTRIES list (e.g. list
        # was edited) -- safest fallback is to just process everything.
        print(f"  (note: {last_country!r} not found in COUNTRIES list -- processing all countries)")
        return countries

    remaining = countries[resume_index:]
    print(f"Resuming from {last_country!r}: {len(remaining)} countries left to process.")
    return remaining


def run_and_save(countries, filepath="tariff_results.csv"):
    remaining = prepare_resume(filepath, countries)
    file_exists = os.path.exists(filepath)
    mode = "a" if file_exists else "w"

    with open(filepath, mode, newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Country", "Utility"] + CATEGORIES)
            f.flush()

        for country in remaining:
            print(f"Processing {country}")

            try:
                utilities = get_utilities(country)
            except Exception as e:
                print(f"  error getting utilities for {country}: {e}")
                continue

            for util in utilities:
                row_values = []
                for cat in CATEGORIES:
                    try:
                        val = get_category_value(country, util, cat)
                        row_values.append(val)
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"  error on {country}/{util}/{cat}: {e}")
                        row_values.append(None)

                writer.writerow([country, util] + row_values)
                f.flush()  # ensures the row is on disk immediately, not buffered

    print(f"Done. Saved {filepath}")


if __name__ == "__main__":
    run_and_save(COUNTRIES)