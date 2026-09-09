"""
point_to_vlm — Data Collection Script
======================================
Records RealSense D435i depth+RGB streams to external SSD (E:\)
and logs all session metadata + QA pairs to an Excel file.

HOW TO USE:
1. Plug in RealSense D435i via USB
2. Plug in external SSD (E:\)
3. Run: python collect_data.py
4. Follow the prompts
5. Press SPACE to start recording, SPACE again to stop
6. Enter QA pairs when prompted
7. Run validator after each room

REQUIREMENTS:
    pip install pyrealsense2 openpyxl

FOLDER STRUCTURE CREATED ON E:\:
    E:\point_to_vlm\
        recordings\         <- .bag files saved here
        data\               <- processed .npy point clouds go here later
        point_to_vlm_log.xlsx  <- Excel log of all sessions
"""

import pyrealsense2 as rs
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import os
import sys
import datetime
import uuid
import time

# ─────────────────────────────────────────────
# CONFIGURATION — change these if needed
# ─────────────────────────────────────────────
SSD_ROOT        = r"E:\point_to_vlm"
RECORDINGS_DIR  = os.path.join(SSD_ROOT, "recordings")
DATA_DIR        = os.path.join(SSD_ROOT, "data")
EXCEL_LOG_PATH  = os.path.join(SSD_ROOT, "point_to_vlm_log.xlsx")

MIN_QA_PAIRS    = 3   # minimum QA pairs per room (mini dataset mode)
TARGET_QA_PAIRS = 5   # target for full dataset

# ─────────────────────────────────────────────
# EXCEL SETUP
# ─────────────────────────────────────────────
HEADERS = [
    "Session ID",
    "Date",
    "Time",
    "Room ID",
    "Room Type",
    "Location",
    "Room Size",
    "Lighting",
    "Approx Object Count",
    "Bag File Path",
    "Recording Duration (s)",
    "Validator Result",
    "Q1", "A1",
    "Q2", "A2",
    "Q3", "A3",
    "Q4", "A4",
    "Q5", "A5",
    "Notes",
]

HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT  = Font(bold=True, color="FFFFFF", name="Arial", size=11)
DATA_FONT    = Font(name="Arial", size=10)
PASS_FILL    = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL    = PatternFill("solid", fgColor="FFC7CE")


def init_excel():
    """Create the Excel log file with headers if it doesn't exist."""
    if os.path.exists(EXCEL_LOG_PATH):
        wb = openpyxl.load_workbook(EXCEL_LOG_PATH)
        ws = wb.active
        return wb, ws

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "point_to_vlm Sessions"

    for col_idx, header in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    # Set column widths
    col_widths = {
        "A": 20, "B": 12, "C": 10, "D": 12, "E": 15,
        "F": 12, "G": 10, "H": 10, "I": 8,  "J": 45,
        "K": 10, "L": 10,
        "M": 35, "N": 25,  # Q1, A1
        "O": 35, "P": 25,  # Q2, A2
        "Q": 35, "R": 25,  # Q3, A3
        "S": 35, "T": 25,  # Q4, A4
        "U": 35, "V": 25,  # Q5, A5
        "W": 30,            # Notes
    }
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    ws.row_dimensions[1].height = 30
    wb.save(EXCEL_LOG_PATH)
    return wb, ws


def log_session(session_data):
    """Append one session row to the Excel log."""
    wb = openpyxl.load_workbook(EXCEL_LOG_PATH)
    ws = wb.active

    next_row = ws.max_row + 1

    row_values = [
        session_data["session_id"],
        session_data["date"],
        session_data["time"],
        session_data["room_id"],
        session_data["room_type"],
        session_data["location"],
        session_data["room_size"],
        session_data["lighting"],
        session_data["object_count"],
        session_data["bag_path"],
        session_data["duration"],
        session_data["validator"],
    ]

    # QA pairs (up to 5)
    for i in range(5):
        if i < len(session_data["qa_pairs"]):
            row_values.append(session_data["qa_pairs"][i]["question"])
            row_values.append(session_data["qa_pairs"][i]["answer"])
        else:
            row_values.append("")
            row_values.append("")

    row_values.append(session_data["notes"])

    for col_idx, value in enumerate(row_values, start=1):
        cell = ws.cell(row=next_row, column=col_idx, value=value)
        cell.font = DATA_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    # Color the validator cell
    validator_col = HEADERS.index("Validator Result") + 1
    v_cell = ws.cell(row=next_row, column=validator_col)
    if session_data["validator"] == "PASS":
        v_cell.fill = PASS_FILL
        v_cell.font = Font(name="Arial", size=10, bold=True, color="375623")
    elif session_data["validator"] == "FAIL":
        v_cell.fill = FAIL_FILL
        v_cell.font = Font(name="Arial", size=10, bold=True, color="9C0006")

    ws.row_dimensions[next_row].height = 60
    wb.save(EXCEL_LOG_PATH)
    print(f"\n  Session logged to Excel: {EXCEL_LOG_PATH}")


# ─────────────────────────────────────────────
# USER INPUT HELPERS
# ─────────────────────────────────────────────
def prompt_choice(prompt, choices):
    """Ask user to pick from a numbered list."""
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    while True:
        try:
            idx = int(input("  Enter number: ").strip())
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(choices)}")


def prompt_text(prompt, allow_empty=False):
    """Ask user for a text input."""
    while True:
        val = input(f"\n{prompt}: ").strip()
        if val or allow_empty:
            return val
        print("  This field cannot be empty.")


def prompt_int(prompt, min_val=0, max_val=999):
    """Ask user for an integer input."""
    while True:
        try:
            val = int(input(f"\n{prompt}: ").strip())
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"  Please enter a number between {min_val} and {max_val}.")


def collect_metadata():
    """Collect all session metadata from the user via prompts."""
    print("\n" + "="*60)
    print("  point_to_vlm — Session Metadata")
    print("="*60)

    now = datetime.datetime.now()

    room_id = prompt_text("Room ID (e.g. room_001, room_002)")
    # Validate format loosely
    if not room_id.startswith("room_"):
        print(f"  Tip: standard format is room_001, room_002 etc. You entered: {room_id}")

    room_type = prompt_choice("Room Type", [
        "bedroom", "living room", "office", "lab",
        "kitchen", "classroom", "corridor", "other"
    ])

    location = prompt_choice("Location", ["home", "IUB", "other"])

    room_size = prompt_choice("Room Size", ["small", "medium", "large"])

    lighting = prompt_choice("Lighting Condition", ["good", "dim", "bright"])

    object_count = prompt_int("Approximate number of visible objects (e.g. 5)", 1, 50)

    return {
        "session_id"   : str(uuid.uuid4())[:8].upper(),
        "date"         : now.strftime("%Y-%m-%d"),
        "time"         : now.strftime("%H:%M:%S"),
        "room_id"      : room_id,
        "room_type"    : room_type,
        "location"     : location,
        "room_size"    : room_size,
        "lighting"     : lighting,
        "object_count" : object_count,
        "bag_path"     : "",
        "duration"     : 0,
        "validator"    : "PENDING",
        "qa_pairs"     : [],
        "notes"        : "",
    }


def collect_qa_pairs(room_id):
    """Collect QA pairs for this room interactively."""
    print(f"\n{'='*60}")
    print(f"  QA Pairs for {room_id}")
    print(f"  Minimum: {MIN_QA_PAIRS} pairs")
    print(f"  Ask ONLY about objects clearly visible in your recording.")
    print(f"  Good question types:")
    print(f"    - What is the large object against the left wall?")
    print(f"    - How many chairs are visible?")
    print(f"    - What color is the object near the window?")
    print(f"    - Is there anything on the table?")
    print(f"    - What is in the corner of the room?")
    print("="*60)

    qa_pairs = []
    pair_num = 1

    while True:
        print(f"\n  QA Pair {pair_num}:")
        question = prompt_text(f"  Question {pair_num}")
        answer   = prompt_text(f"  Answer {pair_num}")
        qa_pairs.append({"question": question, "answer": answer})
        pair_num += 1

        if pair_num > MIN_QA_PAIRS:
            if pair_num > TARGET_QA_PAIRS:
                print(f"\n  You have entered {TARGET_QA_PAIRS} pairs (target reached).")
                break
            add_more = input(f"\n  Add another QA pair? (y/n): ").strip().lower()
            if add_more != "y":
                break

    return qa_pairs


# ─────────────────────────────────────────────
# REALSENSE RECORDING
# ─────────────────────────────────────────────
def record_bag(bag_path):
    """
    Record a RealSense bag file. Press SPACE to start, SPACE again to stop.
    Returns duration in seconds, or 0 if recording failed.
    """
    print(f"\n{'='*60}")
    print(f"  Starting RealSense recording")
    print(f"  Saving to: {bag_path}")
    print(f"  - Hold camera at chest height, both hands")
    print(f"  - Point slightly downward toward the room center")
    print(f"  - Stand STILL during recording")
    print(f"  - Record for 10-15 seconds minimum")
    print(f"\n  Press ENTER to start recording...")
    input()

    pipeline = rs.pipeline()
    config   = rs.config()

    # Enable depth and color streams at standard resolutions
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Enable bag file recording
    config.enable_record_to_file(bag_path)

    try:
        pipeline.start(config)
        print(f"\n  RECORDING... Press ENTER to stop.")
        start_time = time.time()

        input()  # Wait for user to press Enter to stop

        duration = round(time.time() - start_time, 1)
        pipeline.stop()

        print(f"\n  Recording stopped. Duration: {duration}s")
        print(f"  Saved: {bag_path}")
        return duration

    except Exception as e:
        print(f"\n  ERROR during recording: {e}")
        print(f"  Check that the RealSense camera is connected via USB.")
        try:
            pipeline.stop()
        except:
            pass
        return 0


# ─────────────────────────────────────────────
# MAIN COLLECTION FLOW
# ─────────────────────────────────────────────
def main():
    # Setup directories
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Initialize Excel log
    init_excel()
    print(f"\n  Excel log ready: {EXCEL_LOG_PATH}")

    print("\n" + "="*60)
    print("  point_to_vlm Data Collection Tool")
    print("  Project: Frozen Projection Alignment for 3D QA")
    print("="*60)
    print(f"  Recordings will save to: {RECORDINGS_DIR}")
    print(f"  Log file: {EXCEL_LOG_PATH}")

    while True:
        print("\n\n" + "="*60)
        action = prompt_choice("What do you want to do?", [
            "Record a new room",
            "Update validator result for last session",
            "Exit"
        ])

        if action == "Exit":
            print("\n  Exiting. All data saved to SSD.")
            break

        elif action == "Record a new room":
            # Step 1: Collect metadata
            session = collect_metadata()

            # Step 2: Build bag file path
            bag_filename = f"{session['room_id']}_{session['session_id']}.bag"
            bag_path = os.path.join(RECORDINGS_DIR, bag_filename)
            session["bag_path"] = bag_path

            # Step 3: Record
            duration = record_bag(bag_path)
            session["duration"] = duration

            if duration == 0:
                print("\n  Recording failed. Session not logged.")
                continue

            # Step 4: Collect QA pairs
            session["qa_pairs"] = collect_qa_pairs(session["room_id"])

            # Step 5: Validator result
            print(f"\n{'='*60}")
            print(f"  Now run the data validator on this recording.")
            print(f"  Bag file: {bag_path}")
            print(f"  (Run scripts/step8_data_validator.py separately)")
            validator_result = prompt_choice(
                "What did the validator report?",
                ["PASS", "FAIL", "NOT RUN YET"]
            )
            session["validator"] = validator_result

            # Step 6: Notes
            session["notes"] = prompt_text(
                "Any notes about this room? (glass surfaces, dark areas, etc.) [press Enter to skip]",
                allow_empty=True
            )

            # Step 7: Log to Excel
            log_session(session)

            print(f"\n  Session {session['session_id']} complete.")
            print(f"  Room: {session['room_id']} | Validator: {validator_result}")
            print(f"  QA Pairs: {len(session['qa_pairs'])}")

            if validator_result == "FAIL":
                print(f"\n  WARNING: Validator FAILED. Re-record this room before continuing.")
            elif validator_result == "PASS":
                print(f"\n  Good capture. Move to the next room when ready.")

        elif action == "Update validator result for last session":
            wb = openpyxl.load_workbook(EXCEL_LOG_PATH)
            ws = wb.active
            last_row = ws.max_row
            if last_row <= 1:
                print("  No sessions logged yet.")
                continue
            room_id_val = ws.cell(row=last_row, column=HEADERS.index("Room ID") + 1).value
            print(f"\n  Last session: {room_id_val} (row {last_row})")
            new_result = prompt_choice("New validator result", ["PASS", "FAIL"])
            validator_col = HEADERS.index("Validator Result") + 1
            v_cell = ws.cell(row=last_row, column=validator_col, value=new_result)
            if new_result == "PASS":
                v_cell.fill = PASS_FILL
                v_cell.font = Font(name="Arial", size=10, bold=True, color="375623")
            else:
                v_cell.fill = FAIL_FILL
                v_cell.font = Font(name="Arial", size=10, bold=True, color="9C0006")
            wb.save(EXCEL_LOG_PATH)
            print(f"  Updated to {new_result}.")


if __name__ == "__main__":
    main()
