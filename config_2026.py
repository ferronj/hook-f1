"""
Shared constants for 2026 F1 season simulations.

Contains driver lineup, constructor info, team colors, and race calendar.
"""

# 2026 drivers: driverId -> (constructorId, name, abbreviation)
DRIVERS_2026 = {
    846: (1, "Lando Norris", "NOR"),
    857: (1, "Oscar Piastri", "PIA"),
    1:   (6, "Lewis Hamilton", "HAM"),
    844: (6, "Charles Leclerc", "LEC"),
    847: (131, "George Russell", "RUS"),
    863: (131, "Kimi Antonelli", "ANT"),
    830: (9, "Max Verstappen", "VER"),
    864: (9, "Isack Hadjar", "HAD"),
    4:   (117, "Fernando Alonso", "ALO"),
    840: (117, "Lance Stroll", "STR"),
    832: (3, "Carlos Sainz", "SAI"),
    848: (3, "Alex Albon", "ALB"),
    861: (214, "Franco Colapinto", "COL"),
    842: (214, "Pierre Gasly", "GAS"),
    839: (210, "Esteban Ocon", "OCO"),
    860: (210, "Oliver Bearman", "BEA"),
    807: (15, "Nico Hulkenberg", "HUL"),
    865: (15, "Gabriel Bortoleto", "BOR"),
    859: (215, "Liam Lawson", "LAW"),
    866: (215, "Arvid Lindblad", "LIN"),
    822: (216, "Valtteri Bottas", "BOT"),
    815: (216, "Sergio Perez", "PER"),
}

CONSTRUCTOR_NAMES = {
    1: "McLaren", 6: "Ferrari", 131: "Mercedes", 9: "Red Bull",
    117: "Aston Martin", 3: "Williams", 214: "Alpine",
    210: "Haas", 15: "Audi", 215: "Racing Bulls", 216: "Cadillac",
}

TEAM_COLORS = {
    "McLaren": "#FF8000", "Ferrari": "#E8002D", "Mercedes": "#27F4D2",
    "Red Bull": "#3671C6", "Aston Martin": "#229971", "Williams": "#64C4FF",
    "Alpine": "#0093CC", "Haas": "#B6BABD", "Audi": "#00594F",
    "Racing Bulls": "#6692FF", "Cadillac": "#1B3D2F",
}

# 2026 calendar: round -> {name, circuit, circuit_id, date}
# Circuit IDs match Ergast circuits.csv; 81 = new Madrid circuit
# Note: Bahrain and Saudi Arabian Grands Prix were cancelled; Miami onward shifted down by 2.
CALENDAR_2026 = {
    1:  {"name": "Australian Grand Prix", "circuit": "Albert Park, Melbourne", "circuit_id": 1, "date": "2026-03-08"},
    2:  {"name": "Chinese Grand Prix", "circuit": "Shanghai International Circuit", "circuit_id": 17, "date": "2026-03-15"},
    3:  {"name": "Japanese Grand Prix", "circuit": "Suzuka Circuit", "circuit_id": 22, "date": "2026-03-29"},
    4:  {"name": "Miami Grand Prix", "circuit": "Miami International Autodrome", "circuit_id": 79, "date": "2026-05-03"},
    5:  {"name": "Canadian Grand Prix", "circuit": "Circuit Gilles Villeneuve, Montreal", "circuit_id": 7, "date": "2026-05-24"},
    6:  {"name": "Monaco Grand Prix", "circuit": "Circuit de Monaco", "circuit_id": 6, "date": "2026-06-07"},
    7:  {"name": "Barcelona-Catalunya Grand Prix", "circuit": "Circuit de Barcelona-Catalunya", "circuit_id": 4, "date": "2026-06-14"},
    8:  {"name": "Austrian Grand Prix", "circuit": "Red Bull Ring, Spielberg", "circuit_id": 70, "date": "2026-06-28"},
    9:  {"name": "British Grand Prix", "circuit": "Silverstone Circuit", "circuit_id": 9, "date": "2026-07-05"},
    10: {"name": "Belgian Grand Prix", "circuit": "Circuit de Spa-Francorchamps", "circuit_id": 13, "date": "2026-07-19"},
    11: {"name": "Hungarian Grand Prix", "circuit": "Hungaroring, Budapest", "circuit_id": 11, "date": "2026-07-26"},
    12: {"name": "Dutch Grand Prix", "circuit": "Circuit Park Zandvoort", "circuit_id": 39, "date": "2026-08-23"},
    13: {"name": "Italian Grand Prix", "circuit": "Autodromo Nazionale di Monza", "circuit_id": 14, "date": "2026-09-06"},
    14: {"name": "Spanish Grand Prix", "circuit": "Madring, Madrid", "circuit_id": 81, "date": "2026-09-13"},
    15: {"name": "Azerbaijan Grand Prix", "circuit": "Baku City Circuit", "circuit_id": 73, "date": "2026-09-26"},
    16: {"name": "Singapore Grand Prix", "circuit": "Marina Bay Street Circuit", "circuit_id": 15, "date": "2026-10-11"},
    17: {"name": "United States Grand Prix", "circuit": "Circuit of the Americas, Austin", "circuit_id": 69, "date": "2026-10-25"},
    18: {"name": "Mexican Grand Prix", "circuit": "Autodromo Hermanos Rodriguez", "circuit_id": 32, "date": "2026-11-01"},
    19: {"name": "Brazilian Grand Prix", "circuit": "Autodromo Jose Carlos Pace, Sao Paulo", "circuit_id": 18, "date": "2026-11-08"},
    20: {"name": "Las Vegas Grand Prix", "circuit": "Las Vegas Strip Street Circuit", "circuit_id": 80, "date": "2026-11-22"},
    21: {"name": "Qatar Grand Prix", "circuit": "Losail International Circuit", "circuit_id": 78, "date": "2026-11-29"},
    22: {"name": "Abu Dhabi Grand Prix", "circuit": "Yas Marina Circuit", "circuit_id": 24, "date": "2026-12-06"},
}

# Slugs for output file naming
_RACE_SLUGS = {
    1: "australia", 2: "china", 3: "japan", 4: "miami",
    5: "canada", 6: "monaco", 7: "barcelona", 8: "austria",
    9: "britain", 10: "belgium", 11: "hungary", 12: "netherlands",
    13: "italy", 14: "spain", 15: "azerbaijan", 16: "singapore",
    17: "usa", 18: "mexico", 19: "brazil", 20: "las_vegas",
    21: "qatar", 22: "abu_dhabi",
}


def race_slug(round_num):
    """Return a short slug for file naming, e.g. 1 -> 'australia'."""
    return _RACE_SLUGS.get(round_num, f"r{round_num}")
