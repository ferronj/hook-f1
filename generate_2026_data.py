"""
Generate 2026 F1 season data in the same CSV format as the existing Ergast-style dataset.
Appends to existing CSV files (leaving original data untouched).

Checks which rounds already exist before appending. Safe to run multiple times.
Add race results to RACE_RESULTS dict as the season progresses.
"""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ============================================================
# ID offsets (max IDs after 2025 data)
# ============================================================
RACE_ID_START = 1169        # existing max = 1168
RESULT_ID_START = 27244     # existing max = 27243
QUALIFY_ID_START = 10552    # existing max = 10551

# ============================================================
# New entities for 2026
# ============================================================
NEW_DRIVERS = [
    # driverId, driverRef, number, code, forename, surname, dob, nationality, url
    (866, "lindblad", 36, "LIN", "Arvid", "Lindblad", "2006-10-17", "British",
     "http://en.wikipedia.org/wiki/Arvid_Lindblad"),
]

NEW_CONSTRUCTORS = [
    # constructorId, constructorRef, name, nationality, url
    (216, "cadillac", "Cadillac", "American",
     "http://en.wikipedia.org/wiki/Cadillac_F1"),
]

# ============================================================
# Constructor and driver ID mappings for 2026
# ============================================================
CONSTRUCTOR_MAP = {
    "McLaren": 1,
    "Williams": 3,
    "Ferrari": 6,
    "Red Bull Racing": 9,
    "Audi": 15,
    "Aston Martin": 117,
    "Mercedes": 131,
    "Haas F1 Team": 210,
    "Alpine": 214,
    "Racing Bulls": 215,
    "Cadillac": 216,
}

DRIVER_MAP = {
    "Lando Norris": 846,
    "Oscar Piastri": 857,
    "Lewis Hamilton": 1,
    "Charles Leclerc": 844,
    "George Russell": 847,
    "Kimi Antonelli": 863,
    "Max Verstappen": 830,
    "Isack Hadjar": 864,
    "Fernando Alonso": 4,
    "Lance Stroll": 840,
    "Carlos Sainz": 832,
    "Alexander Albon": 848,
    "Franco Colapinto": 861,
    "Pierre Gasly": 842,
    "Esteban Ocon": 839,
    "Oliver Bearman": 860,
    "Nico Hulkenberg": 807,
    "Gabriel Bortoleto": 865,
    "Liam Lawson": 859,
    "Arvid Lindblad": 866,
    "Valtteri Bottas": 822,
    "Sergio Perez": 815,
}

DRIVER_NUM = {
    "Lando Norris": 4,
    "Oscar Piastri": 81,
    "Lewis Hamilton": 44,
    "Charles Leclerc": 16,
    "George Russell": 63,
    "Kimi Antonelli": 12,
    "Max Verstappen": 1,
    "Isack Hadjar": 6,
    "Fernando Alonso": 14,
    "Lance Stroll": 18,
    "Carlos Sainz": 55,
    "Alexander Albon": 23,
    "Franco Colapinto": 43,
    "Pierre Gasly": 10,
    "Esteban Ocon": 31,
    "Oliver Bearman": 87,
    "Nico Hulkenberg": 27,
    "Gabriel Bortoleto": 5,
    "Liam Lawson": 30,
    "Arvid Lindblad": 36,
    "Valtteri Bottas": 77,
    "Sergio Perez": 11,
}

# Circuit mapping (from config_2026.py calendar)
CIRCUIT_MAP = {
    "Australian Grand Prix": 1,
    "Chinese Grand Prix": 17,
    "Japanese Grand Prix": 22,
    "Bahrain Grand Prix": 3,
    "Saudi Arabian Grand Prix": 77,
    "Miami Grand Prix": 79,
    "Canadian Grand Prix": 7,
    "Monaco Grand Prix": 6,
    "Barcelona-Catalunya Grand Prix": 4,
    "Austrian Grand Prix": 70,
    "British Grand Prix": 9,
    "Belgian Grand Prix": 13,
    "Hungarian Grand Prix": 11,
    "Dutch Grand Prix": 39,
    "Italian Grand Prix": 14,
    "Spanish Grand Prix": 81,
    "Azerbaijan Grand Prix": 73,
    "Singapore Grand Prix": 15,
    "United States Grand Prix": 69,
    "Mexican Grand Prix": 32,
    "Brazilian Grand Prix": 18,
    "Las Vegas Grand Prix": 80,
    "Qatar Grand Prix": 78,
    "Abu Dhabi Grand Prix": 24,
}

# ============================================================
# 2026 Race Calendar
# ============================================================
RACES_2026 = [
    # (round, name, date, time)
    (1, "Australian Grand Prix", "2026-03-08", "04:00:00"),
    (2, "Chinese Grand Prix", "2026-03-15", "07:00:00"),
    (3, "Japanese Grand Prix", "2026-03-29", "05:00:00"),
    (4, "Bahrain Grand Prix", "2026-04-12", "15:00:00"),
    (5, "Saudi Arabian Grand Prix", "2026-04-19", "17:00:00"),
    (6, "Miami Grand Prix", "2026-05-03", "20:00:00"),
    (7, "Canadian Grand Prix", "2026-05-24", "18:00:00"),
    (8, "Monaco Grand Prix", "2026-06-07", "13:00:00"),
    (9, "Barcelona-Catalunya Grand Prix", "2026-06-14", "13:00:00"),
    (10, "Austrian Grand Prix", "2026-06-28", "13:00:00"),
    (11, "British Grand Prix", "2026-07-05", "14:00:00"),
    (12, "Belgian Grand Prix", "2026-07-19", "13:00:00"),
    (13, "Hungarian Grand Prix", "2026-07-26", "13:00:00"),
    (14, "Dutch Grand Prix", "2026-08-23", "13:00:00"),
    (15, "Italian Grand Prix", "2026-09-06", "13:00:00"),
    (16, "Spanish Grand Prix", "2026-09-13", "13:00:00"),
    (17, "Azerbaijan Grand Prix", "2026-09-26", "11:00:00"),
    (18, "Singapore Grand Prix", "2026-10-11", "12:00:00"),
    (19, "United States Grand Prix", "2026-10-25", "19:00:00"),
    (20, "Mexican Grand Prix", "2026-11-01", "20:00:00"),
    (21, "Brazilian Grand Prix", "2026-11-08", "17:00:00"),
    (22, "Las Vegas Grand Prix", "2026-11-22", "06:00:00"),
    (23, "Qatar Grand Prix", "2026-11-29", "16:00:00"),
    (24, "Abu Dhabi Grand Prix", "2026-12-06", "13:00:00"),
]

# ============================================================
# 2026 Qualifying Results
# Each entry: (driver_name, constructor_name, grid_position)
# ============================================================
QUALIFYING_RESULTS = {
    1: [  # Australian Grand Prix
        ("George Russell", "Mercedes", 1),
        ("Kimi Antonelli", "Mercedes", 2),
        ("Isack Hadjar", "Red Bull Racing", 3),
        ("Charles Leclerc", "Ferrari", 4),
        ("Oscar Piastri", "McLaren", 5),
        ("Lando Norris", "McLaren", 6),
        ("Lewis Hamilton", "Ferrari", 7),
        ("Liam Lawson", "Racing Bulls", 8),
        ("Arvid Lindblad", "Racing Bulls", 9),
        ("Gabriel Bortoleto", "Audi", 10),
        ("Nico Hulkenberg", "Audi", 11),
        ("Oliver Bearman", "Haas F1 Team", 12),
        ("Esteban Ocon", "Haas F1 Team", 13),
        ("Pierre Gasly", "Alpine", 14),
        ("Alexander Albon", "Williams", 15),
        ("Franco Colapinto", "Alpine", 16),
        ("Fernando Alonso", "Aston Martin", 17),
        ("Sergio Perez", "Cadillac", 18),
        ("Valtteri Bottas", "Cadillac", 19),
        ("Max Verstappen", "Red Bull Racing", 20),
        ("Carlos Sainz", "Williams", 21),
        ("Lance Stroll", "Aston Martin", 22),
    ],
}

# ============================================================
# 2026 Race Results
# Each entry: (driver_name, constructor_name, position, positionText, laps, time_str, points, statusId)
# statusId: 1=Finished, 130=DNF, 20=DNS
# Add results after each race weekend.
# ============================================================
RACE_RESULTS = {
    1: [  # Australian Grand Prix — PLACEHOLDER (update with actual results)
        ("George Russell", "Mercedes", 1, "1", 58, "1:28:00.000", 25, 1),
        ("Kimi Antonelli", "Mercedes", 2, "2", 58, "+3.5", 18, 1),
        ("Oscar Piastri", "McLaren", 3, "3", 58, "+8.2", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 58, "+12.1", 12, 1),
        ("Lando Norris", "McLaren", 5, "5", 58, "+15.6", 10, 1),
        ("Lewis Hamilton", "Ferrari", 6, "6", 58, "+20.3", 8, 1),
        ("Isack Hadjar", "Red Bull Racing", 7, "7", 58, "+25.7", 6, 1),
        ("Liam Lawson", "Racing Bulls", 8, "8", 58, "+30.1", 4, 1),
        ("Max Verstappen", "Red Bull Racing", 9, "9", 58, "+35.2", 2, 1),
        ("Gabriel Bortoleto", "Audi", 10, "10", 58, "+40.5", 1, 1),
        ("Nico Hulkenberg", "Audi", 11, "11", 58, "+42.8", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 12, "12", 58, "+45.1", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 13, "13", 58, "+48.2", 0, 1),
        ("Pierre Gasly", "Alpine", 14, "14", 58, "+50.5", 0, 1),
        ("Alexander Albon", "Williams", 15, "15", 58, "+53.7", 0, 1),
        ("Franco Colapinto", "Alpine", 16, "16", 58, "+55.9", 0, 1),
        ("Fernando Alonso", "Aston Martin", 17, "17", 57, "+1 Lap", 0, 1),
        ("Arvid Lindblad", "Racing Bulls", 18, "18", 57, "+1 Lap", 0, 1),
        ("Sergio Perez", "Cadillac", 19, "19", 57, "+1 Lap", 0, 1),
        ("Valtteri Bottas", "Cadillac", 20, "R", 40, "\\N", 0, 130),
        ("Carlos Sainz", "Williams", 21, "R", 30, "\\N", 0, 130),
        ("Lance Stroll", "Aston Martin", 22, "R", 15, "\\N", 0, 130),
    ],
}


# ============================================================
# Helper functions
# ============================================================
def append_csv(filename, rows):
    """Append rows to a CSV file."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def get_existing_2026_rounds():
    """Check which 2026 rounds already exist in races.csv."""
    filepath = os.path.join(DATA_DIR, "races.csv")
    existing_rounds = set()
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) > 1 and row[1] == "2026":
                existing_rounds.add(int(row[2]))
    return existing_rounds


def main():
    existing_rounds = get_existing_2026_rounds()
    if existing_rounds:
        print(f"2026 rounds already in data: {sorted(existing_rounds)}")

    rounds_with_results = sorted(RACE_RESULTS.keys())
    # Include rounds that have at least qualifying data
    all_rounds = sorted(
        set(QUALIFYING_RESULTS.keys()) | set(RACE_RESULTS.keys())
    )
    new_rounds = [r for r in all_rounds if r not in existing_rounds]

    # Check for rounds that exist but are missing results
    existing_result_races = set()
    results_filepath = os.path.join(DATA_DIR, "results.csv")
    with open(results_filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            existing_result_races.add(int(row[1]))

    rounds_needing_results = [
        r for r in existing_rounds
        if r in RACE_RESULTS and (RACE_ID_START + r - 1) not in existing_result_races
    ]

    rounds_to_add = sorted(set(new_rounds) | set(rounds_needing_results))

    if not rounds_to_add and not rounds_needing_results:
        print("No new 2026 data to add.")
        return

    print(f"Rounds to process: {rounds_to_add}")

    # ================================================================
    # 1. Add new drivers
    # ================================================================
    driver_filepath = os.path.join(DATA_DIR, "drivers.csv")
    existing_driver_ids = set()
    with open(driver_filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            existing_driver_ids.add(int(row[0]))

    driver_rows = []
    for d in NEW_DRIVERS:
        if d[0] not in existing_driver_ids:
            driver_rows.append(d)
            print(f"  Adding new driver: {d[4]} {d[5]} (ID {d[0]})")

    if driver_rows:
        append_csv("drivers.csv", driver_rows)

    # ================================================================
    # 2. Add new constructors
    # ================================================================
    constructor_filepath = os.path.join(DATA_DIR, "constructors.csv")
    existing_constructor_ids = set()
    with open(constructor_filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            existing_constructor_ids.add(int(row[0]))

    constructor_rows = []
    for c in NEW_CONSTRUCTORS:
        if c[0] not in existing_constructor_ids:
            constructor_rows.append(c)
            print(f"  Adding new constructor: {c[2]} (ID {c[0]})")

    if constructor_rows:
        append_csv("constructors.csv", constructor_rows)

    # ================================================================
    # 3. Add races
    # ================================================================
    print("Adding 2026 races...")
    race_rows = []
    for rnd, name, date, time in RACES_2026:
        if rnd in new_rounds:
            race_id = RACE_ID_START + rnd - 1
            circuit_id = CIRCUIT_MAP.get(name, 1)
            url = f"https://en.wikipedia.org/wiki/2026_{name.replace(' ', '_')}"
            race_rows.append((
                race_id, 2026, rnd, circuit_id, name, date, time, url,
                "\\N", "\\N", "\\N", "\\N", "\\N", "\\N", "\\N", "\\N", "\\N", "\\N"
            ))

    if race_rows:
        append_csv("races.csv", race_rows)

    # ================================================================
    # 4. Add race results (for rounds that have results)
    # ================================================================
    result_id = RESULT_ID_START
    result_rows = []

    for rnd in rounds_to_add:
        if rnd not in RACE_RESULTS:
            continue

        race_id = RACE_ID_START + rnd - 1
        results = RACE_RESULTS[rnd]

        # Get grid positions from qualifying
        quali = QUALIFYING_RESULTS.get(rnd, [])
        grid_map = {name: pos for name, _, pos in quali}

        for entry in results:
            driver_name, constructor_name, pos, pos_text, laps, time_str, points, status_id = entry
            did = DRIVER_MAP[driver_name]
            cid = CONSTRUCTOR_MAP[constructor_name]
            num = DRIVER_NUM[driver_name]
            grid = grid_map.get(driver_name, pos)

            result_rows.append((
                result_id, race_id, did, cid, num,
                grid, pos, pos_text, pos,
                points, laps, time_str,
                "\\N", "\\N", "\\N", "\\N", "\\N",
                status_id
            ))
            result_id += 1

    if result_rows:
        print(f"  Added {len(result_rows)} race results")
        append_csv("results.csv", result_rows)

    # ================================================================
    # 5. Add qualifying results
    # ================================================================
    qualify_id = QUALIFY_ID_START
    qualify_rows = []

    for rnd in new_rounds:
        if rnd not in QUALIFYING_RESULTS:
            continue

        race_id = RACE_ID_START + rnd - 1
        quali = QUALIFYING_RESULTS[rnd]

        for driver_name, constructor_name, grid_pos in quali:
            did = DRIVER_MAP[driver_name]
            cid = CONSTRUCTOR_MAP[constructor_name]
            num = DRIVER_NUM[driver_name]

            qualify_rows.append((
                qualify_id, race_id, did, cid, num,
                grid_pos, "\\N", "\\N", "\\N"
            ))
            qualify_id += 1

    if qualify_rows:
        print(f"  Added {len(qualify_rows)} qualifying results")
        append_csv("qualifying.csv", qualify_rows)

    print(f"\nDone! Added {len(rounds_to_add)} round(s) of 2026 data.")
    if rounds_with_results:
        print(f"  Race results available for rounds: {rounds_with_results}")
    else:
        print("  No race results yet — add to RACE_RESULTS dict after each race.")


if __name__ == "__main__":
    main()
