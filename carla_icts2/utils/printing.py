import os
import time

import numpy as np
from prettytable import PrettyTable


def format_value(value):
    """Format value for display in the table, handling enum types."""
    if hasattr(value, "name"):
        return value.name
    return str(value)


def create_data_table(data_list, start_idx=0, max_rows=None):
    """
    Create a pretty table from a list of iteration data.

    Args:
        data_list: List of dictionaries containing DBN data
        start_idx: Starting index for the iteration counter
        max_rows: Maximum number of rows to display

    Returns:
        PrettyTable object
    """
    if not data_list:
        table = PrettyTable(["No Data Available"])
        return table

    # Limit number of rows if specified
    if max_rows and len(data_list) > max_rows:
        data_list = data_list[-max_rows:]

    # Include only non-raw fields (exclude raw values used for calculations)
    fields = [field for field in data_list[0].keys() if not field.endswith("_raw")]

    # Create table with iteration counter and data fields
    table = PrettyTable()
    table.field_names = ["Iter"] + fields

    # Add rows
    for i, data in enumerate(data_list):
        row = [start_idx + i]
        for field in fields:
            row.append(format_value(data.get(field, "N/A")))
        table.add_row(row)

    # Format table
    table.align = "l"  # Left align text
    table.max_width = 160  # Maximum width of the table

    return table


def display_simulation_data(ep_data, episode_num, save_dir="./tables"):
    """
    Display and save a summary of the simulation data.

    Args:
        ep_data: List of data dictionaries from the current episode
        episode_num: Episode number
        save_dir: Directory to save table files
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create tables
    summary_table = create_data_table(ep_data, max_rows=10)

    # Print summary to console
    print(f"\n===== Episode {episode_num} Summary (showing last 10 iterations) =====")
    print(summary_table)

    # Save full table to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    full_table = create_data_table(ep_data)
    filename = os.path.join(save_dir, f"episode_{episode_num}_{timestamp}.txt")

    with open(filename, "w") as f:
        f.write(f"Episode {episode_num} - Full Data ({len(ep_data)} iterations)\n")
        f.write(str(full_table))

    print(f"Full table saved to {filename}")

    # Calculate and display statistics
    print(f"\n----- Statistics for Episode {episode_num} -----")

    # Count state occurrences
    icr_ped_counts = {}
    sn_ped_counts = {}
    ssec_counts = {}

    for data in ep_data:
        # ICR_ped stats
        icr_ped = format_value(data.get("ICR_ped", "Unknown"))
        icr_ped_counts[icr_ped] = icr_ped_counts.get(icr_ped, 0) + 1

        # SN_ped stats
        sn_ped = format_value(data.get("SN_ped", "Unknown"))
        sn_ped_counts[sn_ped] = sn_ped_counts.get(sn_ped, 0) + 1

        # SSEC stats
        ssec = format_value(data.get("SSEC", "Unknown"))
        ssec_counts[ssec] = ssec_counts.get(ssec, 0) + 1

    # Print stats
    print("Pedestrian Intention Counts:")
    for icr, count in sorted(icr_ped_counts.items()):
        percentage = (count / len(ep_data)) * 100
        print(f"  {icr}: {count} iterations ({percentage:.1f}%)")

    print("\nPedestrian Strategy Counts:")
    for sn, count in sorted(sn_ped_counts.items()):
        percentage = (count / len(ep_data)) * 100
        print(f"  {sn}: {count} iterations ({percentage:.1f}%)")

    print("\nSense of Security Counts:")
    for ssec, count in sorted(ssec_counts.items()):
        percentage = (count / len(ep_data)) * 100
        print(f"  {ssec}: {count} iterations ({percentage:.1f}%)")

    print("\n" + "=" * 60)


def display_iteration_data(data, iteration):
    """
    Display data for a single iteration in a compact table.

    Args:
        data: Dictionary containing the current iteration data
        iteration: Current iteration number
    """
    # Create a small table for the current iteration
    table = PrettyTable()
    table.field_names = ["Variable", "Value"]

    # Add car data
    table.add_row(["Iteration", iteration])
    table.add_row(["", ""])
    table.add_row(["CAR DATA", ""])
    table.add_row(["Approaching (A)", format_value(data.get("A_car", "N/A"))])
    table.add_row(["Wheel Stance (WS)", format_value(data.get("WS_car", "N/A"))])
    table.add_row(["Car Body Orientation (CBO)", format_value(data.get("CBO_car", "N/A"))])
    table.add_row(["Acceleration (ACC)", format_value(data.get("ACC_car", "N/A"))])
    table.add_row(["Speed (S)", f"{data.get('speed_car_raw', 0):.1f} m/s ({data.get('S_car', 'N/A')})"])
    table.add_row(["ICR", format_value(data.get("ICR_car", "N/A"))])
    table.add_row(["SN", format_value(data.get("SN_car", "N/A"))])

    # Add pedestrian data
    table.add_row(["", ""])
    table.add_row(["PEDESTRIAN DATA", ""])
    table.add_row(["BO", format_value(data.get("BO_ped", "N/A"))])
    table.add_row(["HO", format_value(data.get("HO_ped", "N/A"))])
    table.add_row(["HIO", format_value(data.get("HIO_ped", "N/A"))])
    table.add_row(["Approaching (A)", format_value(data.get("A_ped", "N/A"))])
    table.add_row(["Acceleration (ACC)", format_value(data.get("ACC_ped", "N/A"))])
    table.add_row(["Speed (S)", f"{data.get('speed_ped_raw', 0):.1f} m/s ({data.get('S_ped', 'N/A')})"])
    table.add_row(["SSEC", format_value(data.get("SSEC", "N/A"))])
    table.add_row(["ICR", format_value(data.get("ICR_ped", "N/A"))])
    table.add_row(["SN", format_value(data.get("SN_ped", "N/A"))])

    # Add interaction data
    table.add_row(["", ""])
    table.add_row(["COMMON DATA", ""])
    table.add_row(["Distance (D)", f"{data.get('D_raw', 0):.1f} m ({data.get('D', 'N/A')})"])

    table.align = "l"
    print(f"\n----- Iteration {iteration} Data -----")
    print(table)
