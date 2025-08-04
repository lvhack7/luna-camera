# in api-server/app/reporting.py
import os
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timedelta
from .database import engine  # Use the synchronous engine for scripts
from .telegram_bot import send_telegram_alert # We can reuse this for reports

REPORTS_DIR = "reports"

def generate_daily_report():
    """
    Connects to the DB, generates analytics for the last 24 hours,
    and saves the report as a CSV.
    """
    print("Generating daily report...")
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    # --- Database Connection ---
    db = engine.connect()
    
    # --- Time Range ---
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    # --- 1. Peak Occupancy Analysis ---
    occupancy_query = text(f"""
        SELECT timestamp, total_occupancy 
        FROM occupancy_logs
        WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
    """)
    df_occupancy = pd.read_sql(occupancy_query, db)
    
    peak_info = "Not enough data"
    if not df_occupancy.empty:
        peak_row = df_occupancy.loc[df_occupancy['total_occupancy'].idxmax()]
        peak_guests = int(peak_row['total_occupancy'])
        peak_time = pd.to_datetime(peak_row['timestamp']).strftime('%H:%M')
        peak_info = f"{peak_guests} guests around {peak_time}"

    # --- 2. Conversion Rate Analysis ---
    transitions_query = text(f"""
        SELECT from_zone, to_zone, tracker_id
        FROM transition_events
        WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
    """)
    df_transitions = pd.read_sql(transitions_query, db)

    total_entries = 0
    queue_to_hall = 0
    conversion_rate = 0.0

    if not df_transitions.empty:
        # Count unique people entering from outside
        total_entries = df_transitions[df_transitions['from_zone'] == 'outside']['tracker_id'].nunique()
        
        # Count unique people who moved from queue to hall1 or hall2
        queue_to_hall = df_transitions[
            (df_transitions['from_zone'] == 'queue') & 
            (df_transitions['to_zone'].isin(['hall1', 'hall2']))
        ]['tracker_id'].nunique()
        
        if total_entries > 0:
            # A simple conversion: how many who entered also went from queue to hall
            # A more accurate one would track the journey of each ID. This is a good start.
            conversion_rate = (queue_to_hall / total_entries) * 100 if total_entries > 0 else 0
    
    # --- 3. Barista Alert Analysis ---
    alerts_query = text(f"""
        SELECT COUNT(*) as alert_count
        FROM alert_logs
        WHERE alert_type = 'STAFF_ABSENCE' AND timestamp BETWEEN '{start_time}' AND '{end_time}'
    """)
    alert_count_result = db.execute(alerts_query).fetchone()
    barista_alerts = alert_count_result[0] if alert_count_result else 0
    
    db.close()

    # --- 4. Assemble and Save Report ---
    report_data = {
        "Metric": ["Total Guests Entered", "Queue-to-Hall Transitions", "Conversion Rate (%)", "Peak Occupancy", "Barista Absence Alerts"],
        "Value": [total_entries, queue_to_hall, f"{conversion_rate:.2f}", peak_info, barista_alerts]
    }
    df_report = pd.DataFrame(report_data)
    
    report_date = start_time.strftime('%Y-%m-%d')
    report_filename = f"{REPORTS_DIR}/daily_summary_{report_date}.csv"
    df_report.to_csv(report_filename, index=False)
    
    print(f"Report saved to {report_filename}")

    # --- 5. Push Report via Telegram ---
    report_summary_text = (
        f"📊 Daily Report for {report_date} 📊\n\n"
        f"- Total Guests Entered: {total_entries}\n"
        f"- Queue-to-Hall Transitions: {queue_to_hall}\n"
        f"- Conversion Rate: {conversion_rate:.2f}%\n"
        f"- Peak Occupancy: {peak_info}\n"
        f"- Barista Absence Alerts: {barista_alerts}\n"
    )
    # Using asyncio.run to call our async telegram function from a sync context
    asyncio.run(send_telegram_alert(report_summary_text))

    return report_filename