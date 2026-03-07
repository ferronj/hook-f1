"""
Generate 2025 F1 season data in the same CSV format as the existing Ergast-style dataset.
Appends to existing CSV files (leaving original data untouched).
"""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ============================================================
# ID offsets (max IDs from existing dataset)
# ============================================================
RACE_ID_START = 1145        # existing max = 1144
RESULT_ID_START = 26765     # existing max = 26764
QUALIFY_ID_START = 10552    # existing max = 10551
CONSTRUCTOR_RESULTS_ID_START = 17130  # existing max = 17129
CONSTRUCTOR_STANDINGS_ID_START = 28983  # existing max = 28982
DRIVER_STANDINGS_ID_START = 73271      # existing max = 73270
SPRINT_RESULT_ID_START = 361           # existing max = 360

# ============================================================
# New drivers not in the existing dataset
# ============================================================
NEW_DRIVERS = [
    # driverId, driverRef, number, code, forename, surname, dob, nationality, url
    (863, "antonelli", 12, "ANT", "Kimi", "Antonelli", "2006-08-25", "Italian",
     "http://en.wikipedia.org/wiki/Kimi_Antonelli"),
    (864, "hadjar", 6, "HAD", "Isack", "Hadjar", "2004-09-28", "French",
     "http://en.wikipedia.org/wiki/Isack_Hadjar"),
    (865, "bortoleto", 5, "BOR", "Gabriel", "Bortoleto", "2004-10-14", "Brazilian",
     "http://en.wikipedia.org/wiki/Gabriel_Bortoleto"),
]

# ============================================================
# Constructor mapping for 2025 (reuse existing IDs)
# "Racing Bulls" was "RB F1 Team" id=215; "Kick Sauber" is Sauber id=15
# ============================================================
CONSTRUCTOR_MAP = {
    "McLaren": 1,
    "Williams": 3,
    "Ferrari": 6,
    "Red Bull Racing": 9,
    "Kick Sauber": 15,
    "Aston Martin": 117,
    "Mercedes": 131,
    "Haas F1 Team": 210,
    "Alpine": 214,
    "Racing Bulls": 215,
}

# Driver ID mapping
DRIVER_MAP = {
    "Lando Norris": 846,
    "Oscar Piastri": 857,
    "Max Verstappen": 830,
    "Charles Leclerc": 844,
    "Lewis Hamilton": 1,
    "George Russell": 847,
    "Kimi Antonelli": 863,
    "Alexander Albon": 848,
    "Carlos Sainz": 832,
    "Fernando Alonso": 4,
    "Nico Hulkenberg": 807,
    "Isack Hadjar": 864,
    "Oliver Bearman": 860,
    "Liam Lawson": 859,
    "Esteban Ocon": 839,
    "Lance Stroll": 840,
    "Yuki Tsunoda": 852,
    "Pierre Gasly": 842,
    "Gabriel Bortoleto": 865,
    "Franco Colapinto": 861,
    "Jack Doohan": 862,
}

# Driver number mapping
DRIVER_NUM = {
    "Lando Norris": 4,
    "Oscar Piastri": 81,
    "Max Verstappen": 1,
    "Charles Leclerc": 16,
    "Lewis Hamilton": 44,
    "George Russell": 63,
    "Kimi Antonelli": 12,
    "Alexander Albon": 23,
    "Carlos Sainz": 55,
    "Fernando Alonso": 14,
    "Nico Hulkenberg": 27,
    "Isack Hadjar": 6,
    "Oliver Bearman": 87,
    "Liam Lawson": 30,
    "Esteban Ocon": 31,
    "Lance Stroll": 18,
    "Yuki Tsunoda": 22,
    "Pierre Gasly": 10,
    "Gabriel Bortoleto": 5,
    "Franco Colapinto": 43,
    "Jack Doohan": 7,
}

# Circuit mapping
CIRCUIT_MAP = {
    "Australian Grand Prix": 1,        # Albert Park
    "Chinese Grand Prix": 17,          # Shanghai
    "Japanese Grand Prix": 22,         # Suzuka
    "Bahrain Grand Prix": 3,           # Bahrain
    "Saudi Arabian Grand Prix": 77,    # Jeddah
    "Miami Grand Prix": 79,            # Miami
    "Emilia Romagna Grand Prix": 21,   # Imola
    "Monaco Grand Prix": 6,            # Monaco
    "Spanish Grand Prix": 4,           # Catalunya
    "Canadian Grand Prix": 7,          # Villeneuve
    "Austrian Grand Prix": 70,         # Red Bull Ring
    "British Grand Prix": 9,           # Silverstone
    "Belgian Grand Prix": 13,          # Spa
    "Hungarian Grand Prix": 11,        # Hungaroring
    "Dutch Grand Prix": 39,            # Zandvoort
    "Italian Grand Prix": 14,          # Monza
    "Azerbaijan Grand Prix": 73,       # Baku
    "Singapore Grand Prix": 15,        # Marina Bay
    "United States Grand Prix": 69,    # COTA
    "Mexico City Grand Prix": 32,      # Rodriguez
    "São Paulo Grand Prix": 18,        # Interlagos
    "Las Vegas Grand Prix": 44,        # Las Vegas
    "Qatar Grand Prix": 78,            # Losail
    "Abu Dhabi Grand Prix": 24,        # Yas Marina
}

# ============================================================
# 2025 Race Calendar
# ============================================================
RACES_2025 = [
    # (round, name, date, time, fp1_date, fp1_time, fp2_date, fp2_time, fp3_date, fp3_time, quali_date, quali_time, sprint_date, sprint_time)
    (1, "Australian Grand Prix", "2025-03-16", "04:00:00", "2025-03-14", "01:30:00", "2025-03-14", "05:00:00", "2025-03-15", "01:30:00", "2025-03-15", "05:00:00", "\\N", "\\N"),
    (2, "Chinese Grand Prix", "2025-03-23", "07:00:00", "2025-03-21", "03:30:00", "2025-03-21", "07:00:00", "2025-03-22", "03:30:00", "2025-03-22", "07:00:00", "\\N", "\\N"),
    (3, "Japanese Grand Prix", "2025-04-06", "05:00:00", "2025-04-04", "02:30:00", "2025-04-04", "06:00:00", "2025-04-05", "02:30:00", "2025-04-05", "06:00:00", "\\N", "\\N"),
    (4, "Bahrain Grand Prix", "2025-04-13", "15:00:00", "2025-04-11", "11:30:00", "2025-04-11", "15:00:00", "2025-04-12", "12:30:00", "2025-04-12", "16:00:00", "\\N", "\\N"),
    (5, "Saudi Arabian Grand Prix", "2025-04-20", "17:00:00", "2025-04-18", "13:30:00", "2025-04-18", "17:00:00", "2025-04-19", "13:30:00", "2025-04-19", "17:00:00", "\\N", "\\N"),
    (6, "Miami Grand Prix", "2025-05-04", "20:00:00", "2025-05-02", "18:30:00", "2025-05-02", "22:00:00", "2025-05-03", "17:00:00", "2025-05-03", "21:00:00", "\\N", "\\N"),
    (7, "Emilia Romagna Grand Prix", "2025-05-18", "13:00:00", "2025-05-16", "11:30:00", "2025-05-16", "15:00:00", "2025-05-17", "10:30:00", "2025-05-17", "14:00:00", "\\N", "\\N"),
    (8, "Monaco Grand Prix", "2025-05-25", "13:00:00", "2025-05-23", "11:30:00", "2025-05-23", "15:00:00", "2025-05-24", "10:30:00", "2025-05-24", "14:00:00", "\\N", "\\N"),
    (9, "Spanish Grand Prix", "2025-06-01", "13:00:00", "2025-05-30", "11:30:00", "2025-05-30", "15:00:00", "2025-05-31", "10:30:00", "2025-05-31", "14:00:00", "\\N", "\\N"),
    (10, "Canadian Grand Prix", "2025-06-15", "18:00:00", "2025-06-13", "17:30:00", "2025-06-13", "21:00:00", "2025-06-14", "16:30:00", "2025-06-14", "20:00:00", "\\N", "\\N"),
    (11, "Austrian Grand Prix", "2025-06-29", "13:00:00", "2025-06-27", "11:30:00", "2025-06-27", "15:00:00", "2025-06-28", "10:30:00", "2025-06-28", "14:00:00", "\\N", "\\N"),
    (12, "British Grand Prix", "2025-07-06", "14:00:00", "2025-07-04", "11:30:00", "2025-07-04", "15:00:00", "2025-07-05", "10:30:00", "2025-07-05", "14:00:00", "\\N", "\\N"),
    (13, "Belgian Grand Prix", "2025-07-27", "13:00:00", "2025-07-25", "11:30:00", "2025-07-25", "15:00:00", "2025-07-26", "10:30:00", "2025-07-26", "14:00:00", "\\N", "\\N"),
    (14, "Hungarian Grand Prix", "2025-08-03", "13:00:00", "2025-08-01", "11:30:00", "2025-08-01", "15:00:00", "2025-08-02", "10:30:00", "2025-08-02", "14:00:00", "\\N", "\\N"),
    (15, "Dutch Grand Prix", "2025-08-31", "13:00:00", "2025-08-29", "10:30:00", "2025-08-29", "14:00:00", "2025-08-30", "10:30:00", "2025-08-30", "14:00:00", "\\N", "\\N"),
    (16, "Italian Grand Prix", "2025-09-07", "13:00:00", "2025-09-05", "11:30:00", "2025-09-05", "15:00:00", "2025-09-06", "10:30:00", "2025-09-06", "14:00:00", "\\N", "\\N"),
    (17, "Azerbaijan Grand Prix", "2025-09-21", "11:00:00", "2025-09-19", "09:30:00", "2025-09-19", "13:00:00", "2025-09-20", "09:30:00", "2025-09-20", "13:00:00", "\\N", "\\N"),
    (18, "Singapore Grand Prix", "2025-10-05", "12:00:00", "2025-10-03", "09:30:00", "2025-10-03", "13:00:00", "2025-10-04", "09:30:00", "2025-10-04", "14:00:00", "\\N", "\\N"),
    (19, "United States Grand Prix", "2025-10-19", "19:00:00", "2025-10-17", "17:30:00", "2025-10-17", "21:00:00", "2025-10-18", "19:00:00", "2025-10-18", "22:00:00", "\\N", "\\N"),
    (20, "Mexico City Grand Prix", "2025-10-26", "20:00:00", "2025-10-24", "18:30:00", "2025-10-24", "22:00:00", "2025-10-25", "17:30:00", "2025-10-25", "21:00:00", "\\N", "\\N"),
    (21, "São Paulo Grand Prix", "2025-11-09", "17:00:00", "2025-11-07", "14:30:00", "2025-11-07", "18:00:00", "2025-11-08", "14:30:00", "2025-11-08", "18:00:00", "\\N", "\\N"),
    (22, "Las Vegas Grand Prix", "2025-11-22", "06:00:00", "2025-11-21", "02:30:00", "2025-11-21", "06:00:00", "2025-11-22", "02:30:00", "2025-11-22", "06:00:00", "\\N", "\\N"),
    (23, "Qatar Grand Prix", "2025-11-30", "16:00:00", "2025-11-28", "14:30:00", "2025-11-28", "18:00:00", "2025-11-29", "15:00:00", "2025-11-29", "18:00:00", "\\N", "\\N"),
    (24, "Abu Dhabi Grand Prix", "2025-12-07", "13:00:00", "2025-12-05", "09:30:00", "2025-12-05", "13:00:00", "2025-12-06", "10:30:00", "2025-12-06", "14:00:00", "\\N", "\\N"),
]

# ============================================================
# 2025 Race Results - Full classifications for all 24 races
# Each entry: (driver_name, constructor_name, position, positionText, laps, time_str, points, statusId)
# statusId: 1=Finished, 2=Disqualified, 3=Accident, 4=Collision, 130=Retired, 131=Mechanical, 20=DNS
# For simplicity: 1=Finished, 130=DNF-generic, 20=DNS, 2=DSQ
# ============================================================

RACE_RESULTS = {
    1: [  # Australian Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 57, "1:42:06.304", 25, 1),
        ("Max Verstappen", "Red Bull Racing", 2, "2", 57, "+0.895", 18, 1),
        ("George Russell", "Mercedes", 3, "3", 57, "+8.481", 15, 1),
        ("Kimi Antonelli", "Mercedes", 4, "4", 57, "+10.135", 12, 1),
        ("Alexander Albon", "Williams", 5, "5", 57, "+12.773", 10, 1),
        ("Lance Stroll", "Aston Martin", 6, "6", 57, "+17.413", 8, 1),
        ("Nico Hulkenberg", "Kick Sauber", 7, "7", 57, "+18.423", 6, 1),
        ("Charles Leclerc", "Ferrari", 8, "8", 57, "+19.826", 4, 1),
        ("Oscar Piastri", "McLaren", 9, "9", 57, "+20.448", 2, 1),
        ("Lewis Hamilton", "Ferrari", 10, "10", 57, "+22.473", 1, 1),
        ("Pierre Gasly", "Alpine", 11, "11", 57, "+26.502", 0, 1),
        ("Yuki Tsunoda", "Racing Bulls", 12, "12", 57, "+29.884", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 13, "13", 57, "+33.161", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 14, "14", 57, "+40.351", 0, 1),
        ("Liam Lawson", "Red Bull Racing", 15, "R", 46, "\\N", 0, 130),
        ("Gabriel Bortoleto", "Kick Sauber", 16, "R", 45, "\\N", 0, 130),
        ("Fernando Alonso", "Aston Martin", 17, "R", 32, "\\N", 0, 130),
        ("Carlos Sainz", "Williams", 18, "R", 0, "\\N", 0, 130),
        ("Jack Doohan", "Alpine", 19, "R", 0, "\\N", 0, 130),
        ("Isack Hadjar", "Racing Bulls", 20, "D", 0, "\\N", 0, 20),
    ],
    2: [  # Chinese Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 56, "1:30:55.026", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 56, "+9.748", 18, 1),
        ("George Russell", "Mercedes", 3, "3", 56, "+11.097", 15, 1),
        ("Max Verstappen", "Red Bull Racing", 4, "4", 56, "+16.656", 12, 1),
        ("Esteban Ocon", "Haas F1 Team", 5, "5", 56, "+49.969", 10, 1),
        ("Kimi Antonelli", "Mercedes", 6, "6", 56, "+53.748", 8, 1),
        ("Alexander Albon", "Williams", 7, "7", 56, "+56.321", 6, 1),
        ("Oliver Bearman", "Haas F1 Team", 8, "8", 56, "+61.303", 4, 1),
        ("Lance Stroll", "Aston Martin", 9, "9", 56, "+70.204", 2, 1),
        ("Carlos Sainz", "Williams", 10, "10", 56, "+76.387", 1, 1),
        ("Isack Hadjar", "Racing Bulls", 11, "11", 56, "+78.875", 0, 1),
        ("Liam Lawson", "Red Bull Racing", 12, "12", 56, "+81.147", 0, 1),
        ("Jack Doohan", "Alpine", 13, "13", 56, "+88.401", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 14, "14", 55, "+1 Lap", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 15, "15", 55, "+1 Lap", 0, 1),
        ("Yuki Tsunoda", "Racing Bulls", 16, "16", 55, "+1 Lap", 0, 1),
        ("Fernando Alonso", "Aston Martin", 17, "R", 4, "\\N", 0, 130),
        ("Charles Leclerc", "Ferrari", 18, "D", 56, "\\N", 0, 2),
        ("Lewis Hamilton", "Ferrari", 19, "D", 56, "\\N", 0, 2),
        ("Pierre Gasly", "Alpine", 20, "D", 56, "\\N", 0, 2),
    ],
    3: [  # Japanese Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 53, "1:22:06.983", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 53, "+1.423", 18, 1),
        ("Oscar Piastri", "McLaren", 3, "3", 53, "+2.129", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 53, "+16.097", 12, 1),
        ("George Russell", "Mercedes", 5, "5", 53, "+17.362", 10, 1),
        ("Kimi Antonelli", "Mercedes", 6, "6", 53, "+18.671", 8, 1),
        ("Lewis Hamilton", "Ferrari", 7, "7", 53, "+29.182", 6, 1),
        ("Isack Hadjar", "Racing Bulls", 8, "8", 53, "+37.134", 4, 1),
        ("Alexander Albon", "Williams", 9, "9", 53, "+40.367", 2, 1),
        ("Oliver Bearman", "Haas F1 Team", 10, "10", 53, "+54.529", 1, 1),
        ("Fernando Alonso", "Aston Martin", 11, "11", 53, "+57.333", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 12, "12", 53, "+58.401", 0, 1),
        ("Pierre Gasly", "Alpine", 13, "13", 53, "+62.122", 0, 1),
        ("Carlos Sainz", "Williams", 14, "14", 53, "+74.129", 0, 1),
        ("Jack Doohan", "Alpine", 15, "15", 53, "+81.314", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 16, "16", 53, "+81.957", 0, 1),
        ("Liam Lawson", "Racing Bulls", 17, "17", 53, "+82.734", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 18, "18", 53, "+83.438", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 19, "19", 53, "+83.897", 0, 1),
        ("Lance Stroll", "Aston Martin", 20, "20", 52, "+1 Lap", 0, 1),
    ],
    4: [  # Bahrain Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 57, "1:35:39.435", 25, 1),
        ("George Russell", "Mercedes", 2, "2", 57, "+15.499", 18, 1),
        ("Lando Norris", "McLaren", 3, "3", 57, "+16.273", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 57, "+19.679", 12, 1),
        ("Lewis Hamilton", "Ferrari", 5, "5", 57, "+27.993", 10, 1),
        ("Max Verstappen", "Red Bull Racing", 6, "6", 57, "+34.395", 8, 1),
        ("Pierre Gasly", "Alpine", 7, "7", 57, "+36.002", 6, 1),
        ("Esteban Ocon", "Haas F1 Team", 8, "8", 57, "+44.244", 4, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 9, "9", 57, "+45.061", 2, 1),
        ("Oliver Bearman", "Haas F1 Team", 10, "10", 57, "+47.594", 1, 1),
        ("Kimi Antonelli", "Mercedes", 11, "11", 57, "+48.016", 0, 1),
        ("Alexander Albon", "Williams", 12, "12", 57, "+48.839", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 13, "13", 57, "+56.314", 0, 1),
        ("Jack Doohan", "Alpine", 14, "14", 57, "+57.806", 0, 1),
        ("Fernando Alonso", "Aston Martin", 15, "15", 57, "+60.340", 0, 1),
        ("Liam Lawson", "Racing Bulls", 16, "16", 57, "+64.435", 0, 1),
        ("Lance Stroll", "Aston Martin", 17, "17", 57, "+65.489", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 18, "18", 57, "+66.872", 0, 1),
        ("Carlos Sainz", "Williams", 19, "R", 45, "\\N", 0, 130),
        ("Nico Hulkenberg", "Kick Sauber", 20, "D", 57, "\\N", 0, 2),
    ],
    5: [  # Saudi Arabian Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 50, "1:21:06.758", 25, 1),
        ("Max Verstappen", "Red Bull Racing", 2, "2", 50, "+2.843", 18, 1),
        ("Charles Leclerc", "Ferrari", 3, "3", 50, "+8.104", 15, 1),
        ("Lando Norris", "McLaren", 4, "4", 50, "+9.196", 12, 1),
        ("George Russell", "Mercedes", 5, "5", 50, "+27.236", 10, 1),
        ("Kimi Antonelli", "Mercedes", 6, "6", 50, "+34.688", 8, 1),
        ("Lewis Hamilton", "Ferrari", 7, "7", 50, "+39.073", 6, 1),
        ("Carlos Sainz", "Williams", 8, "8", 50, "+64.630", 4, 1),
        ("Alexander Albon", "Williams", 9, "9", 50, "+66.515", 2, 1),
        ("Isack Hadjar", "Racing Bulls", 10, "10", 50, "+67.091", 1, 1),
        ("Fernando Alonso", "Aston Martin", 11, "11", 50, "+75.917", 0, 1),
        ("Liam Lawson", "Racing Bulls", 12, "12", 50, "+78.451", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 13, "13", 50, "+79.194", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 14, "14", 50, "+99.723", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 15, "15", 49, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 16, "16", 49, "+1 Lap", 0, 1),
        ("Jack Doohan", "Alpine", 17, "17", 49, "+1 Lap", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 18, "18", 49, "+1 Lap", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 19, "R", 1, "\\N", 0, 130),
        ("Pierre Gasly", "Alpine", 20, "R", 0, "\\N", 0, 130),
    ],
    6: [  # Miami Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 57, "1:28:51.587", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 57, "+4.630", 18, 1),
        ("George Russell", "Mercedes", 3, "3", 57, "+37.644", 15, 1),
        ("Max Verstappen", "Red Bull Racing", 4, "4", 57, "+39.956", 12, 1),
        ("Alexander Albon", "Williams", 5, "5", 57, "+48.067", 10, 1),
        ("Kimi Antonelli", "Mercedes", 6, "6", 57, "+55.502", 8, 1),
        ("Charles Leclerc", "Ferrari", 7, "7", 57, "+57.036", 6, 1),
        ("Lewis Hamilton", "Ferrari", 8, "8", 57, "+60.186", 4, 1),
        ("Carlos Sainz", "Williams", 9, "9", 57, "+60.577", 2, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 10, "10", 57, "+74.434", 1, 1),
        ("Isack Hadjar", "Racing Bulls", 11, "11", 57, "+74.602", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 12, "12", 57, "+82.006", 0, 1),
        ("Pierre Gasly", "Alpine", 13, "13", 57, "+90.445", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 14, "14", 56, "+1 Lap", 0, 1),
        ("Fernando Alonso", "Aston Martin", 15, "15", 56, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 16, "16", 56, "+1 Lap", 0, 1),
        ("Liam Lawson", "Racing Bulls", 17, "R", 36, "\\N", 0, 130),
        ("Gabriel Bortoleto", "Kick Sauber", 18, "R", 30, "\\N", 0, 130),
        ("Oliver Bearman", "Haas F1 Team", 19, "R", 27, "\\N", 0, 130),
        ("Jack Doohan", "Alpine", 20, "R", 0, "\\N", 0, 130),
    ],
    7: [  # Emilia Romagna Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 63, "1:31:33.199", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 63, "+6.109", 18, 1),
        ("Oscar Piastri", "McLaren", 3, "3", 63, "+12.956", 15, 1),
        ("Lewis Hamilton", "Ferrari", 4, "4", 63, "+14.356", 12, 1),
        ("Alexander Albon", "Williams", 5, "5", 63, "+17.945", 10, 1),
        ("Charles Leclerc", "Ferrari", 6, "6", 63, "+20.774", 8, 1),
        ("George Russell", "Mercedes", 7, "7", 63, "+22.034", 6, 1),
        ("Carlos Sainz", "Williams", 8, "8", 63, "+22.898", 4, 1),
        ("Isack Hadjar", "Racing Bulls", 9, "9", 63, "+23.586", 2, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 10, "10", 63, "+26.446", 1, 1),
        ("Fernando Alonso", "Aston Martin", 11, "11", 63, "+27.250", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 12, "12", 63, "+30.296", 0, 1),
        ("Pierre Gasly", "Alpine", 13, "13", 63, "+31.424", 0, 1),
        ("Liam Lawson", "Racing Bulls", 14, "14", 63, "+32.511", 0, 1),
        ("Lance Stroll", "Aston Martin", 15, "15", 63, "+32.993", 0, 1),
        ("Franco Colapinto", "Alpine", 16, "16", 63, "+33.411", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 17, "17", 63, "+33.808", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 18, "18", 63, "+38.572", 0, 1),
        ("Kimi Antonelli", "Mercedes", 19, "R", 44, "\\N", 0, 130),
        ("Esteban Ocon", "Haas F1 Team", 20, "R", 27, "\\N", 0, 130),
    ],
    8: [  # Monaco Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 78, "1:40:33.843", 25, 1),
        ("Charles Leclerc", "Ferrari", 2, "2", 78, "+3.131", 18, 1),
        ("Oscar Piastri", "McLaren", 3, "3", 78, "+3.658", 15, 1),
        ("Max Verstappen", "Red Bull Racing", 4, "4", 78, "+20.572", 12, 1),
        ("Lewis Hamilton", "Ferrari", 5, "5", 78, "+51.387", 10, 1),
        ("Isack Hadjar", "Racing Bulls", 6, "6", 77, "+1 Lap", 8, 1),
        ("Esteban Ocon", "Haas F1 Team", 7, "7", 77, "+1 Lap", 6, 1),
        ("Liam Lawson", "Racing Bulls", 8, "8", 77, "+1 Lap", 4, 1),
        ("Alexander Albon", "Williams", 9, "9", 76, "+2 Laps", 2, 1),
        ("Carlos Sainz", "Williams", 10, "10", 76, "+2 Laps", 1, 1),
        ("George Russell", "Mercedes", 11, "11", 76, "+2 Laps", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 12, "12", 76, "+2 Laps", 0, 1),
        ("Franco Colapinto", "Alpine", 13, "13", 76, "+2 Laps", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 14, "14", 76, "+2 Laps", 0, 1),
        ("Lance Stroll", "Aston Martin", 15, "15", 76, "+2 Laps", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 16, "16", 76, "+2 Laps", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 17, "17", 76, "+2 Laps", 0, 1),
        ("Kimi Antonelli", "Mercedes", 18, "18", 75, "+3 Laps", 0, 1),
        ("Fernando Alonso", "Aston Martin", 19, "R", 36, "\\N", 0, 130),
        ("Pierre Gasly", "Alpine", 20, "R", 7, "\\N", 0, 130),
    ],
    9: [  # Spanish Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 66, "1:32:57.375", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 66, "+2.471", 18, 1),
        ("Charles Leclerc", "Ferrari", 3, "3", 66, "+10.455", 15, 1),
        ("George Russell", "Mercedes", 4, "4", 66, "+11.359", 12, 1),
        ("Nico Hulkenberg", "Kick Sauber", 5, "5", 66, "+13.648", 10, 1),
        ("Lewis Hamilton", "Ferrari", 6, "6", 66, "+15.508", 8, 1),
        ("Isack Hadjar", "Racing Bulls", 7, "7", 66, "+16.022", 6, 1),
        ("Pierre Gasly", "Alpine", 8, "8", 66, "+17.882", 4, 1),
        ("Fernando Alonso", "Aston Martin", 9, "9", 66, "+21.564", 2, 1),
        ("Max Verstappen", "Red Bull Racing", 10, "10", 66, "+21.826", 1, 1),
        ("Liam Lawson", "Racing Bulls", 11, "11", 66, "+25.532", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 12, "12", 66, "+25.996", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 13, "13", 66, "+28.822", 0, 1),
        ("Carlos Sainz", "Williams", 14, "14", 66, "+29.309", 0, 1),
        ("Franco Colapinto", "Alpine", 15, "15", 66, "+31.381", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 16, "16", 66, "+32.197", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 17, "17", 66, "+37.065", 0, 1),
        ("Kimi Antonelli", "Mercedes", 18, "R", 53, "\\N", 0, 130),
        ("Alexander Albon", "Williams", 19, "R", 27, "\\N", 0, 130),
    ],
    10: [  # Canadian Grand Prix
        ("George Russell", "Mercedes", 1, "1", 70, "1:31:52.688", 25, 1),
        ("Max Verstappen", "Red Bull Racing", 2, "2", 70, "+0.228", 18, 1),
        ("Kimi Antonelli", "Mercedes", 3, "3", 70, "+1.014", 15, 1),
        ("Oscar Piastri", "McLaren", 4, "4", 70, "+2.109", 12, 1),
        ("Charles Leclerc", "Ferrari", 5, "5", 70, "+3.442", 10, 1),
        ("Lewis Hamilton", "Ferrari", 6, "6", 70, "+10.713", 8, 1),
        ("Fernando Alonso", "Aston Martin", 7, "7", 70, "+10.972", 6, 1),
        ("Nico Hulkenberg", "Kick Sauber", 8, "8", 70, "+15.364", 4, 1),
        ("Esteban Ocon", "Haas F1 Team", 9, "9", 69, "+1 Lap", 2, 1),
        ("Carlos Sainz", "Williams", 10, "10", 69, "+1 Lap", 1, 1),
        ("Oliver Bearman", "Haas F1 Team", 11, "11", 69, "+1 Lap", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 12, "12", 69, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 13, "13", 69, "+1 Lap", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 14, "14", 69, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 15, "15", 69, "+1 Lap", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 16, "16", 69, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 17, "17", 69, "+1 Lap", 0, 1),
        ("Lando Norris", "McLaren", 18, "R", 66, "\\N", 0, 130),
        ("Liam Lawson", "Racing Bulls", 19, "R", 53, "\\N", 0, 130),
        ("Alexander Albon", "Williams", 20, "R", 46, "\\N", 0, 130),
    ],
    11: [  # Austrian Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 70, "1:23:47.693", 25, 1),
        ("Oscar Piastri", "McLaren", 2, "2", 70, "+2.695", 18, 1),
        ("Charles Leclerc", "Ferrari", 3, "3", 70, "+19.820", 15, 1),
        ("Lewis Hamilton", "Ferrari", 4, "4", 70, "+29.020", 12, 1),
        ("George Russell", "Mercedes", 5, "5", 70, "+62.396", 10, 1),
        ("Liam Lawson", "Racing Bulls", 6, "6", 70, "+67.754", 8, 1),
        ("Fernando Alonso", "Aston Martin", 7, "7", 69, "+1 Lap", 6, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 8, "8", 69, "+1 Lap", 4, 1),
        ("Nico Hulkenberg", "Kick Sauber", 9, "9", 69, "+1 Lap", 2, 1),
        ("Esteban Ocon", "Haas F1 Team", 10, "10", 69, "+1 Lap", 1, 1),
        ("Oliver Bearman", "Haas F1 Team", 11, "11", 69, "+1 Lap", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 12, "12", 69, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 13, "13", 69, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 14, "14", 69, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 15, "15", 69, "+1 Lap", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 16, "16", 68, "+2 Laps", 0, 1),
        ("Alexander Albon", "Williams", 17, "R", 15, "\\N", 0, 130),
        ("Max Verstappen", "Red Bull Racing", 18, "R", 0, "\\N", 0, 130),
        ("Kimi Antonelli", "Mercedes", 19, "R", 0, "\\N", 0, 130),
        ("Carlos Sainz", "Williams", 20, "D", 0, "\\N", 0, 20),
    ],
    12: [  # British Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 52, "1:37:15.735", 25, 1),
        ("Oscar Piastri", "McLaren", 2, "2", 52, "+6.812", 18, 1),
        ("Nico Hulkenberg", "Kick Sauber", 3, "3", 52, "+34.742", 15, 1),
        ("Lewis Hamilton", "Ferrari", 4, "4", 52, "+39.812", 12, 1),
        ("Max Verstappen", "Red Bull Racing", 5, "5", 52, "+56.781", 10, 1),
        ("Pierre Gasly", "Alpine", 6, "6", 52, "+59.857", 8, 1),
        ("Lance Stroll", "Aston Martin", 7, "7", 52, "+60.603", 6, 1),
        ("Alexander Albon", "Williams", 8, "8", 52, "+64.135", 4, 1),
        ("Fernando Alonso", "Aston Martin", 9, "9", 52, "+65.858", 2, 1),
        ("George Russell", "Mercedes", 10, "10", 52, "+70.674", 1, 1),
        ("Oliver Bearman", "Haas F1 Team", 11, "11", 52, "+72.095", 0, 1),
        ("Carlos Sainz", "Williams", 12, "12", 52, "+76.592", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 13, "13", 52, "+77.301", 0, 1),
        ("Charles Leclerc", "Ferrari", 14, "14", 52, "+84.477", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 15, "15", 51, "+1 Lap", 0, 1),
        ("Kimi Antonelli", "Mercedes", 16, "R", 23, "\\N", 0, 130),
        ("Isack Hadjar", "Racing Bulls", 17, "R", 17, "\\N", 0, 130),
        ("Gabriel Bortoleto", "Kick Sauber", 18, "R", 3, "\\N", 0, 130),
        ("Liam Lawson", "Racing Bulls", 19, "R", 0, "\\N", 0, 130),
        ("Franco Colapinto", "Alpine", 20, "D", 0, "\\N", 0, 20),
    ],
    13: [  # Belgian Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 44, "1:25:22.601", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 44, "+3.415", 18, 1),
        ("Charles Leclerc", "Ferrari", 3, "3", 44, "+20.185", 15, 1),
        ("Max Verstappen", "Red Bull Racing", 4, "4", 44, "+21.731", 12, 1),
        ("George Russell", "Mercedes", 5, "5", 44, "+34.863", 10, 1),
        ("Alexander Albon", "Williams", 6, "6", 44, "+39.926", 8, 1),
        ("Lewis Hamilton", "Ferrari", 7, "7", 44, "+40.679", 6, 1),
        ("Liam Lawson", "Racing Bulls", 8, "8", 44, "+52.033", 4, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 9, "9", 44, "+56.434", 2, 1),
        ("Pierre Gasly", "Alpine", 10, "10", 44, "+72.714", 1, 1),
        ("Oliver Bearman", "Haas F1 Team", 11, "11", 44, "+73.145", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 12, "12", 44, "+73.628", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 13, "13", 44, "+75.395", 0, 1),
        ("Lance Stroll", "Aston Martin", 14, "14", 44, "+79.831", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 15, "15", 44, "+86.063", 0, 1),
        ("Kimi Antonelli", "Mercedes", 16, "16", 44, "+86.721", 0, 1),
        ("Fernando Alonso", "Aston Martin", 17, "17", 44, "+87.924", 0, 1),
        ("Carlos Sainz", "Williams", 18, "18", 44, "+92.024", 0, 1),
        ("Franco Colapinto", "Alpine", 19, "19", 44, "+95.250", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 20, "20", 43, "+1 Lap", 0, 1),
    ],
    14: [  # Hungarian Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 70, "1:35:21.231", 25, 1),
        ("Oscar Piastri", "McLaren", 2, "2", 70, "+0.698", 18, 1),
        ("George Russell", "Mercedes", 3, "3", 70, "+21.916", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 70, "+42.560", 12, 1),
        ("Fernando Alonso", "Aston Martin", 5, "5", 70, "+59.040", 10, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 6, "6", 70, "+66.169", 8, 1),
        ("Lance Stroll", "Aston Martin", 7, "7", 70, "+68.174", 6, 1),
        ("Liam Lawson", "Racing Bulls", 8, "8", 70, "+69.451", 4, 1),
        ("Max Verstappen", "Red Bull Racing", 9, "9", 70, "+72.645", 2, 1),
        ("Kimi Antonelli", "Mercedes", 10, "10", 69, "+1 Lap", 1, 1),
        ("Isack Hadjar", "Racing Bulls", 11, "11", 69, "+1 Lap", 0, 1),
        ("Lewis Hamilton", "Ferrari", 12, "12", 69, "+1 Lap", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 13, "13", 69, "+1 Lap", 0, 1),
        ("Carlos Sainz", "Williams", 14, "14", 69, "+1 Lap", 0, 1),
        ("Alexander Albon", "Williams", 15, "15", 69, "+1 Lap", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 16, "16", 69, "+1 Lap", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 17, "17", 69, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 18, "18", 69, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 19, "19", 69, "+1 Lap", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 20, "R", 48, "\\N", 0, 130),
    ],
    15: [  # Dutch Grand Prix
        ("Oscar Piastri", "McLaren", 1, "1", 72, "1:38:29.849", 25, 1),
        ("Max Verstappen", "Red Bull Racing", 2, "2", 72, "+1.271", 18, 1),
        ("Isack Hadjar", "Racing Bulls", 3, "3", 72, "+3.233", 15, 1),
        ("George Russell", "Mercedes", 4, "4", 72, "+5.654", 12, 1),
        ("Alexander Albon", "Williams", 5, "5", 72, "+6.327", 10, 1),
        ("Oliver Bearman", "Haas F1 Team", 6, "6", 72, "+9.044", 8, 1),
        ("Lance Stroll", "Aston Martin", 7, "7", 72, "+9.497", 6, 1),
        ("Fernando Alonso", "Aston Martin", 8, "8", 72, "+11.709", 4, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 9, "9", 72, "+13.597", 2, 1),
        ("Esteban Ocon", "Haas F1 Team", 10, "10", 72, "+14.063", 1, 1),
        ("Franco Colapinto", "Alpine", 11, "11", 72, "+14.511", 0, 1),
        ("Liam Lawson", "Racing Bulls", 12, "12", 72, "+17.063", 0, 1),
        ("Carlos Sainz", "Williams", 13, "13", 72, "+17.376", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 14, "14", 72, "+19.725", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 15, "15", 72, "+21.565", 0, 1),
        ("Kimi Antonelli", "Mercedes", 16, "16", 72, "+22.029", 0, 1),
        ("Pierre Gasly", "Alpine", 17, "17", 72, "+23.629", 0, 1),
        ("Lando Norris", "McLaren", 18, "R", 64, "\\N", 0, 130),
        ("Charles Leclerc", "Ferrari", 19, "R", 52, "\\N", 0, 130),
        ("Lewis Hamilton", "Ferrari", 20, "R", 22, "\\N", 0, 130),
    ],
    16: [  # Italian Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 53, "1:13:24.325", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 53, "+19.207", 18, 1),
        ("Oscar Piastri", "McLaren", 3, "3", 53, "+21.351", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 53, "+25.624", 12, 1),
        ("George Russell", "Mercedes", 5, "5", 53, "+32.881", 10, 1),
        ("Lewis Hamilton", "Ferrari", 6, "6", 53, "+37.449", 8, 1),
        ("Alexander Albon", "Williams", 7, "7", 53, "+50.537", 6, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 8, "8", 53, "+58.484", 4, 1),
        ("Kimi Antonelli", "Mercedes", 9, "9", 53, "+59.762", 2, 1),
        ("Isack Hadjar", "Racing Bulls", 10, "10", 53, "+63.891", 1, 1),
        ("Carlos Sainz", "Williams", 11, "11", 53, "+64.469", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 12, "12", 53, "+79.288", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 13, "13", 53, "+80.701", 0, 1),
        ("Liam Lawson", "Racing Bulls", 14, "14", 53, "+82.351", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 15, "15", 52, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 16, "16", 52, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 17, "17", 52, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 18, "18", 52, "+1 Lap", 0, 1),
        ("Fernando Alonso", "Aston Martin", 19, "R", 24, "\\N", 0, 130),
        ("Nico Hulkenberg", "Kick Sauber", 20, "D", 0, "\\N", 0, 20),
    ],
    17: [  # Azerbaijan Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 51, "1:33:26.408", 25, 1),
        ("George Russell", "Mercedes", 2, "2", 51, "+14.609", 18, 1),
        ("Carlos Sainz", "Williams", 3, "3", 51, "+19.199", 15, 1),
        ("Kimi Antonelli", "Mercedes", 4, "4", 51, "+21.760", 12, 1),
        ("Liam Lawson", "Racing Bulls", 5, "5", 51, "+33.290", 10, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 6, "6", 51, "+33.808", 8, 1),
        ("Lando Norris", "McLaren", 7, "7", 51, "+34.227", 6, 1),
        ("Lewis Hamilton", "Ferrari", 8, "8", 51, "+36.310", 4, 1),
        ("Charles Leclerc", "Ferrari", 9, "9", 51, "+36.774", 2, 1),
        ("Isack Hadjar", "Racing Bulls", 10, "10", 51, "+38.982", 1, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 11, "11", 51, "+67.606", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 12, "12", 51, "+68.262", 0, 1),
        ("Alexander Albon", "Williams", 13, "13", 51, "+72.870", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 14, "14", 51, "+77.580", 0, 1),
        ("Fernando Alonso", "Aston Martin", 15, "15", 51, "+78.707", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 16, "16", 51, "+80.237", 0, 1),
        ("Lance Stroll", "Aston Martin", 17, "17", 51, "+96.392", 0, 1),
        ("Pierre Gasly", "Alpine", 18, "18", 50, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 19, "19", 50, "+1 Lap", 0, 1),
        ("Oscar Piastri", "McLaren", 20, "R", 0, "\\N", 0, 130),
    ],
    18: [  # Singapore Grand Prix
        ("George Russell", "Mercedes", 1, "1", 62, "1:40:22.367", 25, 1),
        ("Max Verstappen", "Red Bull Racing", 2, "2", 62, "+5.430", 18, 1),
        ("Lando Norris", "McLaren", 3, "3", 62, "+6.066", 15, 1),
        ("Oscar Piastri", "McLaren", 4, "4", 62, "+8.146", 12, 1),
        ("Kimi Antonelli", "Mercedes", 5, "5", 62, "+33.681", 10, 1),
        ("Charles Leclerc", "Ferrari", 6, "6", 62, "+45.996", 8, 1),
        ("Fernando Alonso", "Aston Martin", 7, "7", 62, "+80.667", 6, 1),
        ("Lewis Hamilton", "Ferrari", 8, "8", 62, "+85.251", 4, 1),
        ("Oliver Bearman", "Haas F1 Team", 9, "9", 62, "+93.527", 2, 1),
        ("Carlos Sainz", "Williams", 10, "10", 61, "+1 Lap", 1, 1),
        ("Isack Hadjar", "Racing Bulls", 11, "11", 61, "+1 Lap", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 12, "12", 61, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 13, "13", 61, "+1 Lap", 0, 1),
        ("Alexander Albon", "Williams", 14, "14", 61, "+1 Lap", 0, 1),
        ("Liam Lawson", "Racing Bulls", 15, "15", 61, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 16, "16", 61, "+1 Lap", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 17, "17", 61, "+1 Lap", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 18, "18", 61, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 19, "19", 61, "+1 Lap", 0, 1),
        ("Nico Hulkenberg", "Kick Sauber", 20, "20", 61, "+1 Lap", 0, 1),
    ],
    19: [  # United States Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 56, "1:34:00.161", 25, 1),
        ("Lando Norris", "McLaren", 2, "2", 56, "+7.959", 18, 1),
        ("Charles Leclerc", "Ferrari", 3, "3", 56, "+15.373", 15, 1),
        ("Lewis Hamilton", "Ferrari", 4, "4", 56, "+28.536", 12, 1),
        ("Oscar Piastri", "McLaren", 5, "5", 56, "+29.678", 10, 1),
        ("George Russell", "Mercedes", 6, "6", 56, "+33.456", 8, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 7, "7", 56, "+52.714", 6, 1),
        ("Nico Hulkenberg", "Kick Sauber", 8, "8", 56, "+57.249", 4, 1),
        ("Oliver Bearman", "Haas F1 Team", 9, "9", 56, "+64.722", 2, 1),
        ("Fernando Alonso", "Aston Martin", 10, "10", 56, "+70.001", 1, 1),
        ("Liam Lawson", "Racing Bulls", 11, "11", 56, "+73.209", 0, 1),
        ("Lance Stroll", "Aston Martin", 12, "12", 56, "+74.778", 0, 1),
        ("Kimi Antonelli", "Mercedes", 13, "13", 56, "+75.746", 0, 1),
        ("Alexander Albon", "Williams", 14, "14", 56, "+80.000", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 15, "15", 56, "+83.043", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 16, "16", 56, "+92.807", 0, 1),
        ("Franco Colapinto", "Alpine", 17, "17", 55, "+1 Lap", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 18, "18", 55, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 19, "19", 55, "+1 Lap", 0, 1),
        ("Carlos Sainz", "Williams", 20, "R", 5, "\\N", 0, 130),
    ],
    20: [  # Mexico City Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 71, "1:37:58.574", 25, 1),
        ("Charles Leclerc", "Ferrari", 2, "2", 71, "+30.324", 18, 1),
        ("Max Verstappen", "Red Bull Racing", 3, "3", 71, "+31.049", 15, 1),
        ("Oliver Bearman", "Haas F1 Team", 4, "4", 71, "+40.955", 12, 1),
        ("Oscar Piastri", "McLaren", 5, "5", 71, "+42.065", 10, 1),
        ("Kimi Antonelli", "Mercedes", 6, "6", 71, "+47.837", 8, 1),
        ("George Russell", "Mercedes", 7, "7", 71, "+50.287", 6, 1),
        ("Lewis Hamilton", "Ferrari", 8, "8", 71, "+56.446", 4, 1),
        ("Esteban Ocon", "Haas F1 Team", 9, "9", 71, "+75.464", 2, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 10, "10", 71, "+76.863", 1, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 11, "11", 71, "+79.048", 0, 1),
        ("Alexander Albon", "Williams", 12, "12", 70, "+1 Lap", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 13, "13", 70, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 14, "14", 70, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 15, "15", 70, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 16, "16", 70, "+1 Lap", 0, 1),
        ("Carlos Sainz", "Williams", 17, "R", 67, "\\N", 0, 130),
        ("Fernando Alonso", "Aston Martin", 18, "R", 34, "\\N", 0, 130),
        ("Nico Hulkenberg", "Kick Sauber", 19, "R", 25, "\\N", 0, 130),
        ("Liam Lawson", "Racing Bulls", 20, "R", 5, "\\N", 0, 130),
    ],
    21: [  # São Paulo Grand Prix
        ("Lando Norris", "McLaren", 1, "1", 71, "1:32:01.596", 25, 1),
        ("Kimi Antonelli", "Mercedes", 2, "2", 71, "+10.388", 18, 1),
        ("Max Verstappen", "Red Bull Racing", 3, "3", 71, "+10.750", 15, 1),
        ("George Russell", "Mercedes", 4, "4", 71, "+15.267", 12, 1),
        ("Oscar Piastri", "McLaren", 5, "5", 71, "+15.749", 10, 1),
        ("Oliver Bearman", "Haas F1 Team", 6, "6", 71, "+29.630", 8, 1),
        ("Liam Lawson", "Racing Bulls", 7, "7", 71, "+52.642", 6, 1),
        ("Isack Hadjar", "Racing Bulls", 8, "8", 71, "+52.873", 4, 1),
        ("Nico Hulkenberg", "Kick Sauber", 9, "9", 71, "+53.324", 2, 1),
        ("Pierre Gasly", "Alpine", 10, "10", 71, "+53.914", 1, 1),
        ("Alexander Albon", "Williams", 11, "11", 71, "+54.184", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 12, "12", 71, "+54.696", 0, 1),
        ("Carlos Sainz", "Williams", 13, "13", 71, "+55.420", 0, 1),
        ("Fernando Alonso", "Aston Martin", 14, "14", 71, "+55.766", 0, 1),
        ("Franco Colapinto", "Alpine", 15, "15", 71, "+57.777", 0, 1),
        ("Lance Stroll", "Aston Martin", 16, "16", 71, "+58.247", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 17, "17", 71, "+69.176", 0, 1),
        ("Lewis Hamilton", "Ferrari", 18, "R", 37, "\\N", 0, 130),
        ("Charles Leclerc", "Ferrari", 19, "R", 5, "\\N", 0, 130),
        ("Gabriel Bortoleto", "Kick Sauber", 20, "R", 0, "\\N", 0, 130),
    ],
    22: [  # Las Vegas Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 50, "1:21:08.429", 25, 1),
        ("George Russell", "Mercedes", 2, "2", 50, "+23.546", 18, 1),
        ("Kimi Antonelli", "Mercedes", 3, "3", 50, "+30.488", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 50, "+30.678", 12, 1),
        ("Carlos Sainz", "Williams", 5, "5", 50, "+34.924", 10, 1),
        ("Isack Hadjar", "Racing Bulls", 6, "6", 50, "+45.257", 8, 1),
        ("Nico Hulkenberg", "Kick Sauber", 7, "7", 50, "+51.134", 6, 1),
        ("Lewis Hamilton", "Ferrari", 8, "8", 50, "+59.369", 4, 1),
        ("Esteban Ocon", "Haas F1 Team", 9, "9", 50, "+60.635", 2, 1),
        ("Oliver Bearman", "Haas F1 Team", 10, "10", 50, "+70.549", 1, 1),
        ("Fernando Alonso", "Aston Martin", 11, "11", 50, "+85.308", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 12, "12", 50, "+86.974", 0, 1),
        ("Pierre Gasly", "Alpine", 13, "13", 50, "+91.702", 0, 1),
        ("Liam Lawson", "Racing Bulls", 14, "14", 49, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 15, "15", 49, "+1 Lap", 0, 1),
        ("Alexander Albon", "Williams", 16, "R", 35, "\\N", 0, 130),
        ("Gabriel Bortoleto", "Kick Sauber", 17, "R", 2, "\\N", 0, 130),
        ("Lance Stroll", "Aston Martin", 18, "R", 0, "\\N", 0, 130),
        ("Lando Norris", "McLaren", 19, "D", 50, "\\N", 0, 2),
        ("Oscar Piastri", "McLaren", 20, "D", 50, "\\N", 0, 2),
    ],
    23: [  # Qatar Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 57, "1:24:38.241", 25, 1),
        ("Oscar Piastri", "McLaren", 2, "2", 57, "+7.995", 18, 1),
        ("Carlos Sainz", "Williams", 3, "3", 57, "+22.665", 15, 1),
        ("Lando Norris", "McLaren", 4, "4", 57, "+23.315", 12, 1),
        ("Kimi Antonelli", "Mercedes", 5, "5", 57, "+28.317", 10, 1),
        ("George Russell", "Mercedes", 6, "6", 57, "+48.599", 8, 1),
        ("Fernando Alonso", "Aston Martin", 7, "7", 57, "+54.045", 6, 1),
        ("Charles Leclerc", "Ferrari", 8, "8", 57, "+56.785", 4, 1),
        ("Liam Lawson", "Racing Bulls", 9, "9", 57, "+60.073", 2, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 10, "10", 57, "+61.770", 1, 1),
        ("Alexander Albon", "Williams", 11, "11", 57, "+66.931", 0, 1),
        ("Lewis Hamilton", "Ferrari", 12, "12", 57, "+77.730", 0, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 13, "13", 57, "+84.812", 0, 1),
        ("Franco Colapinto", "Alpine", 14, "14", 56, "+1 Lap", 0, 1),
        ("Esteban Ocon", "Haas F1 Team", 15, "15", 56, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 16, "16", 56, "+1 Lap", 0, 1),
        ("Lance Stroll", "Aston Martin", 17, "R", 55, "\\N", 0, 130),
        ("Isack Hadjar", "Racing Bulls", 18, "R", 55, "\\N", 0, 130),
        ("Oliver Bearman", "Haas F1 Team", 19, "R", 41, "\\N", 0, 130),
        ("Nico Hulkenberg", "Kick Sauber", 20, "R", 6, "\\N", 0, 130),
    ],
    24: [  # Abu Dhabi Grand Prix
        ("Max Verstappen", "Red Bull Racing", 1, "1", 58, "1:26:07.469", 25, 1),
        ("Oscar Piastri", "McLaren", 2, "2", 58, "+12.594", 18, 1),
        ("Lando Norris", "McLaren", 3, "3", 58, "+16.572", 15, 1),
        ("Charles Leclerc", "Ferrari", 4, "4", 58, "+23.279", 12, 1),
        ("George Russell", "Mercedes", 5, "5", 58, "+48.563", 10, 1),
        ("Fernando Alonso", "Aston Martin", 6, "6", 58, "+67.562", 8, 1),
        ("Esteban Ocon", "Haas F1 Team", 7, "7", 58, "+69.876", 6, 1),
        ("Lewis Hamilton", "Ferrari", 8, "8", 58, "+72.670", 4, 1),
        ("Nico Hulkenberg", "Kick Sauber", 9, "9", 58, "+79.014", 2, 1),
        ("Lance Stroll", "Aston Martin", 10, "10", 58, "+79.523", 1, 1),
        ("Gabriel Bortoleto", "Kick Sauber", 11, "11", 58, "+81.043", 0, 1),
        ("Oliver Bearman", "Haas F1 Team", 12, "12", 58, "+81.166", 0, 1),
        ("Carlos Sainz", "Williams", 13, "13", 58, "+82.158", 0, 1),
        ("Yuki Tsunoda", "Red Bull Racing", 14, "14", 58, "+83.794", 0, 1),
        ("Kimi Antonelli", "Mercedes", 15, "15", 58, "+84.399", 0, 1),
        ("Alexander Albon", "Williams", 16, "16", 58, "+90.327", 0, 1),
        ("Isack Hadjar", "Racing Bulls", 17, "17", 57, "+1 Lap", 0, 1),
        ("Liam Lawson", "Racing Bulls", 18, "18", 57, "+1 Lap", 0, 1),
        ("Pierre Gasly", "Alpine", 19, "19", 57, "+1 Lap", 0, 1),
        ("Franco Colapinto", "Alpine", 20, "20", 57, "+1 Lap", 0, 1),
    ],
}


def append_csv(filename, rows):
    """Append rows to an existing CSV file."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"  Appended {len(rows)} rows to {filename}")


def generate_all():
    # ================================================================
    # 1. Add new drivers
    # ================================================================
    print("Adding new drivers...")
    driver_rows = []
    for d in NEW_DRIVERS:
        driver_rows.append(d)
    append_csv("drivers.csv", driver_rows)

    # ================================================================
    # 2. Add 2025 season
    # ================================================================
    print("Adding 2025 season...")
    append_csv("seasons.csv", [
        (2025, "http://en.wikipedia.org/wiki/2025_Formula_One_World_Championship")
    ])

    # ================================================================
    # 3. Add races
    # ================================================================
    print("Adding 2025 races...")
    race_rows = []
    for race in RACES_2025:
        rnd, name, date, time, fp1d, fp1t, fp2d, fp2t, fp3d, fp3t, qd, qt, sd, st = race
        race_id = RACE_ID_START + rnd - 1
        circuit_id = CIRCUIT_MAP[name]
        url = f"https://en.wikipedia.org/wiki/2025_{name.replace(' ', '_')}"
        race_rows.append((
            race_id, 2025, rnd, circuit_id, name, date, time, url,
            fp1d, fp1t, fp2d, fp2t, fp3d, fp3t, qd, qt, sd, st
        ))
    append_csv("races.csv", race_rows)

    # ================================================================
    # 4. Add race results
    # ================================================================
    print("Adding 2025 race results...")
    result_rows = []
    result_id = RESULT_ID_START

    for rnd in range(1, 25):
        race_id = RACE_ID_START + rnd - 1
        results = RACE_RESULTS[rnd]

        for entry in results:
            driver_name, constructor_name, pos, pos_text, laps, time_str, points, status_id = entry
            driver_id = DRIVER_MAP[driver_name]
            constructor_id = CONSTRUCTOR_MAP[constructor_name]
            number = DRIVER_NUM[driver_name]

            # Grid position not available from our data - use position as approximation
            grid = pos if pos <= 20 else 0

            # Handle time/milliseconds
            if time_str == "\\N":
                time_val = "\\N"
                milliseconds = "\\N"
            elif time_str.startswith("1:") or time_str.startswith("2:"):
                time_val = time_str
                milliseconds = "\\N"  # Winner's time, could compute but not critical
            elif time_str.startswith("+"):
                time_val = time_str
                milliseconds = "\\N"
            else:
                time_val = time_str
                milliseconds = "\\N"

            result_rows.append((
                result_id, race_id, driver_id, constructor_id, number,
                grid, pos if pos_text not in ("R", "D") else "\\N",
                pos_text, pos, points, laps, time_val, milliseconds,
                "\\N", "\\N", "\\N", "\\N", status_id
            ))
            result_id += 1

    append_csv("results.csv", result_rows)

    # ================================================================
    # 5. Add constructor results (aggregate points per constructor per race)
    # ================================================================
    print("Adding 2025 constructor results...")
    cr_rows = []
    cr_id = CONSTRUCTOR_RESULTS_ID_START

    for rnd in range(1, 25):
        race_id = RACE_ID_START + rnd - 1
        results = RACE_RESULTS[rnd]

        # Aggregate points by constructor
        constructor_points = {}
        for entry in results:
            _, constructor_name, _, _, _, _, points, _ = entry
            cid = CONSTRUCTOR_MAP[constructor_name]
            constructor_points[cid] = constructor_points.get(cid, 0) + points

        for cid, pts in sorted(constructor_points.items()):
            cr_rows.append((cr_id, race_id, cid, pts, "\\N"))
            cr_id += 1

    append_csv("constructor_results.csv", cr_rows)

    # ================================================================
    # 6. Add driver standings (cumulative after each race)
    # ================================================================
    print("Adding 2025 driver standings...")
    ds_rows = []
    ds_id = DRIVER_STANDINGS_ID_START

    # Track cumulative points and wins
    driver_cumulative = {}  # driver_id -> {points, wins}

    for rnd in range(1, 25):
        race_id = RACE_ID_START + rnd - 1
        results = RACE_RESULTS[rnd]

        # Update cumulative stats
        for entry in results:
            driver_name, _, pos, _, _, _, points, _ = entry
            did = DRIVER_MAP[driver_name]
            if did not in driver_cumulative:
                driver_cumulative[did] = {"points": 0, "wins": 0}
            driver_cumulative[did]["points"] += points
            if pos == 1:
                driver_cumulative[did]["wins"] += 1

        # Sort by points (descending) for position
        sorted_drivers = sorted(
            driver_cumulative.items(),
            key=lambda x: (-x[1]["points"], -x[1]["wins"])
        )

        for position, (did, stats) in enumerate(sorted_drivers, 1):
            ds_rows.append((
                ds_id, race_id, did, stats["points"],
                position, str(position), stats["wins"]
            ))
            ds_id += 1

    append_csv("driver_standings.csv", ds_rows)

    # ================================================================
    # 7. Add constructor standings (cumulative after each race)
    # ================================================================
    print("Adding 2025 constructor standings...")
    cs_rows = []
    cs_id = CONSTRUCTOR_STANDINGS_ID_START

    constructor_cumulative = {}  # constructor_id -> {points, wins}

    for rnd in range(1, 25):
        race_id = RACE_ID_START + rnd - 1
        results = RACE_RESULTS[rnd]

        # Track constructor race points and wins
        race_constructor_points = {}
        for entry in results:
            _, constructor_name, pos, _, _, _, points, _ = entry
            cid = CONSTRUCTOR_MAP[constructor_name]
            race_constructor_points[cid] = race_constructor_points.get(cid, 0) + points
            if cid not in constructor_cumulative:
                constructor_cumulative[cid] = {"points": 0, "wins": 0}
            constructor_cumulative[cid]["points"] += points
            if pos == 1:
                constructor_cumulative[cid]["wins"] += 1

        # Sort by points for position
        sorted_constructors = sorted(
            constructor_cumulative.items(),
            key=lambda x: (-x[1]["points"], -x[1]["wins"])
        )

        for position, (cid, stats) in enumerate(sorted_constructors, 1):
            cs_rows.append((
                cs_id, race_id, cid, stats["points"],
                position, str(position), stats["wins"]
            ))
            cs_id += 1

    append_csv("constructor_standings.csv", cs_rows)

    print("\nDone! 2025 season data has been appended to all CSV files.")
    print(f"  - {len(race_rows)} races added")
    print(f"  - {len(result_rows)} race results added")
    print(f"  - {len(cr_rows)} constructor results added")
    print(f"  - {len(ds_rows)} driver standing entries added")
    print(f"  - {len(cs_rows)} constructor standing entries added")
    print(f"  - {len(driver_rows)} new drivers added")


if __name__ == "__main__":
    generate_all()
