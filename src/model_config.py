WESTERN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

INDIAN_INSTRUMENTS = [
    "mridangam",
    "sitar",
    "tabla",
    "veena",
    "violin_indian",
]

TASKS = {
    "family": sorted(WESTERN_GENRES + INDIAN_INSTRUMENTS),
    "western": WESTERN_GENRES,
    "indian": INDIAN_INSTRUMENTS,
}

FAMILY_MAP = {label: "western" for label in WESTERN_GENRES}
FAMILY_MAP.update({label: "indian" for label in INDIAN_INSTRUMENTS})
