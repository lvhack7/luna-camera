import os
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timedelta
import asyncio
from .database import engine
from .telegram_bot import send_telegram_message

REPORTS_DIR = "reports"

def generate_daily_report():
    print("Generating daily report...")
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    
    db = engine.connect()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    # 1. Peak Occupancy
    occ_query = text(f"SELECT timestamp, total_occupancy FROM occupancy_logs WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'")
    df_occ = pd.read_sql(occ_query, db)
    peak_info = "Not enough data"
    if not df_occ.empty:
        peak_row = df_occ.loc[df_occ['total_occupancy'].idxmax()]
        peak_info = f"{int(peak_row['total_occupancy'])} guests at {pd.to_datetime(peak_row['timestamp']).strftime('%H:%M')} UTC"

    # 2. Conversion & Entry/Exit Stats
    trans_query = text(f"SELECT from_zone, to_zone, tracker_id FROM transition_events WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'")
    df_trans = pd.read_sql(trans_query, db)
    total_entries, queue_entries, queue_to_hall = 0, 0, 0
    if not df_trans.empty:
        total_entries = df_trans[(df_trans['from_zone'] == 'outside') & (df_trans['to_zone'] != 'outside')]['tracker_id'].nunique()
        queue_entries = df_trans[df_trans['to_zone'] == 'queue']['tracker_id'].nunique()
        queue_to_hall = df_trans[(df_trans['from_zone'] == 'queue') & (df_trans['to_zone'] == 'hall')]['tracker_id'].nunique()
    conversion_rate = (queue_to_hall / queue_entries) * 100 if queue_entries > 0 else 0

    # 3. Alert Stats
    alert_query = text(f"SELECT COUNT(*) FROM alert_logs WHERE alert_type = 'STAFF_ABSENCE' AND timestamp BETWEEN '{start_time}' AND '{end_time}'")
    barista_alerts = db.execute(alert_query).scalar_one_or_none() or 0
    
    db.close()

    # 4. Assemble and Save Report
    report_data = {
        "Metric": ["Total Unique Entries", "Unique People in Queue", "Queue-to-Hall Transitions", "Queue Conversion Rate (%)", "Peak Occupancy", "Barista Absence Alerts"],
        "Value": [total_entries, queue_entries, queue_to_hall, f"{conversion_rate:.2f}", peak_info, barista_alerts]
    }
    df_report = pd.DataFrame(report_data)
    report_date = start_time.strftime('%Y-%m-%d')
    report_filename = os.path.join(REPORTS_DIR, f"daily_summary_{report_date}.csv")
    df_report.to_csv(report_filename, index=False)
    print(f"Report saved to {report_filename}")

    # 5. Push Report Summary
    summary = (
        f"📊 *Daily Analytics Report: {report_date}*\n\n"
        f"✅ *Entries & Conversion:*\n"
        f"  - Total Guests Entered: `{total_entries}`\n"
        f"  - Entered Queue: `{queue_entries}`\n"
        f"  - Converted (Queue → Hall): `{queue_to_hall}`\n"
        f"  - Conversion Rate: `{conversion_rate:.2f}%`\n\n"
        f"📈 *Occupancy:*\n"
        f"  - Peak Guests: `{peak_info}`\n\n"
        f"🚨 *Alerts:*\n"
        f"  - Barista Absence Alerts: `{barista_alerts}`"
    )
    asyncio.run(send_telegram_message(summary))
    return report_filename