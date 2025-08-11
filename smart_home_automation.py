"""
Smart Home Automation (single-file)
Technologies: Python (Flask, paho-mqtt, APScheduler, SQLite), intended for Raspberry Pi

Features:
- Flask web dashboard to view and control devices
- MQTT client (paho-mqtt) to publish commands and listen to device status
- SQLite database to store devices and scheduled automation tasks
- APScheduler to run scheduled tasks (one-off and daily recurring)

Setup (on Raspberry Pi or any Linux machine):
1) Create a virtual environment (recommended) and install dependencies:
   pip install flask paho-mqtt apscheduler

2) Run the app:
   python smart_home_automation.py

3) Open http://127.0.0.1:5000 in your browser.

Notes / production considerations:
- Secure MQTT with username/password/TLS for internet-exposed brokers.
- Use nginx + gunicorn for production Flask deployment.
- For many devices or high throughput, move to a dedicated message queue/processors and persistent scheduler.

"""

import os
import sqlite3
import json
import threading
from datetime import datetime, time as dtime
from typing import List, Dict, Any

from flask import Flask, request, render_template_string, redirect, url_for, jsonify
import paho.mqtt.client as mqtt
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.cron import CronTrigger

# ----------------------- Configuration -----------------------
DB_PATH = 'smart_home.db'
MQTT_BROKER = os.environ.get('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))
MQTT_KEEPALIVE = 60
MQTT_USERNAME = os.environ.get('MQTT_USERNAME')
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD')

# Topics convention: devices/{device_id}/set  and devices/{device_id}/status
TOPIC_SET = 'devices/{}/set'
TOPIC_STATUS = 'devices/{}/status'

app = Flask(__name__)
scheduler = BackgroundScheduler()
mqtt_client = mqtt.Client()

# ----------------------- Database Utilities -----------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            topic TEXT NOT NULL UNIQUE,
            state TEXT
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            command TEXT NOT NULL,
            schedule_time TEXT NOT NULL, -- ISO 8601 string or HH:MM for daily
            repeat TEXT NOT NULL DEFAULT 'once', -- 'once' or 'daily'
            created_at TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()


def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ----------------------- MQTT Handlers -----------------------


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker {MQTT_BROKER}:{MQTT_PORT} with rc={rc}")
    # Subscribe to status topics wildcard
    client.subscribe('devices/+/status')


def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        print(f"MQTT message {topic}: {payload}")
        # If topic matches devices/{device_id}/status, extract device_id
        parts = topic.split('/')
        if len(parts) >= 3 and parts[0] == 'devices' and parts[2] == 'status':
            device_id = parts[1]
            # Update device state in DB
            conn = db_connection()
            cur = conn.cursor()
            cur.execute('UPDATE devices SET state=? WHERE id=?', (payload, device_id))
            conn.commit()
            conn.close()
    except Exception as e:
        print('Error in on_message:', e)


def setup_mqtt_client():
    if MQTT_USERNAME:
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
    mqtt_client.loop_start()

# ----------------------- Scheduler Utilities -----------------------


def send_mqtt_command(device_id: int, command: str):
    # Look up device topic
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('SELECT topic FROM devices WHERE id=?', (device_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        print(f"Device {device_id} not found for scheduled job")
        return
    topic = row['topic']
    topic_set = TOPIC_SET.format(topic)
    print(f"Publishing to {topic_set}: {command}")
    mqtt_client.publish(topic_set, command)


def schedule_task_job(task_id: int, device_id: int, command: str, schedule_time: str, repeat: str):
    """
    schedule_time: if repeat=='once' -> ISO datetime string (YYYY-MM-DDTHH:MM:SS)
                   if repeat=='daily' -> 'HH:MM' string
    """
    job_id = f"task_{task_id}"
    try:
        # Remove existing job if exists
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
    except Exception:
        pass

    if repeat == 'once':
        dt = datetime.fromisoformat(schedule_time)
        trigger = DateTrigger(run_date=dt)
        scheduler.add_job(send_mqtt_command, trigger=trigger, args=(device_id, command), id=job_id)
        print(f"Scheduled one-off task {task_id} at {dt}")
    elif repeat == 'daily':
        # schedule_time is HH:MM
        hh, mm = map(int, schedule_time.split(':'))
        trigger = CronTrigger(hour=hh, minute=mm)
        scheduler.add_job(send_mqtt_command, trigger=trigger, args=(device_id, command), id=job_id)
        print(f"Scheduled daily task {task_id} at {schedule_time}")
    else:
        print(f"Unknown repeat type: {repeat}")


def load_and_schedule_all_tasks():
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM tasks')
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        schedule_task_job(row['id'], row['device_id'], row['command'], row['schedule_time'], row['repeat'])

# ----------------------- Flask Routes & UI -----------------------

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Home Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 1rem; }
    .container { max-width: 900px; margin: 0 auto; }
    .device { border: 1px solid #ddd; padding: 1rem; margin-bottom: 0.5rem; border-radius: 6px; display:flex; justify-content:space-between; align-items:center}
    .controls { display:flex; gap:8px; }
    form.inline { display:inline; }
    .tasks { margin-top: 1.5rem; }
    table { width:100%; border-collapse: collapse; }
    th, td { padding:8px; border-bottom:1px solid #eee; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Smart Home Dashboard</h1>

    <h2>Devices</h2>
    {% if devices|length == 0 %}
      <p>No devices found. Use the form below to add one.</p>
    {% endif %}
    {% for d in devices %}
      <div class="device">
        <div>
          <strong>{{d['name']}}</strong> <small>(topic: {{d['topic']}})</small>
          <div>State: <em>{{d['state']}}</em></div>
        </div>
        <div class="controls">
          <form class="inline" action="/device/toggle/{{d['id']}}" method="post">
            <input type="hidden" name="command" value="TOGGLE">
            <button type="submit">Toggle</button>
          </form>
          <form class="inline" action="/device/command/{{d['id']}}" method="post">
            <input name="command" placeholder="custom command" required>
            <button type="submit">Send</button>
          </form>
        </div>
      </div>
    {% endfor %}

    <h3>Add device</h3>
    <form action="/device/add" method="post">
      <input name="name" placeholder="Device name" required>
      <input name="topic" placeholder="device topic (unique)" required>
      <button type="submit">Add</button>
    </form>

    <div class="tasks">
      <h2>Scheduled Tasks</h2>
      <table>
        <thead><tr><th>ID</th><th>Device</th><th>Command</th><th>Schedule</th><th>Repeat</th><th>Actions</th></tr></thead>
        <tbody>
        {% for t in tasks %}
          <tr>
            <td>{{t['id']}}</td>
            <td>{{t['device_name']}}</td>
            <td>{{t['command']}}</td>
            <td>{{t['schedule_time']}}</td>
            <td>{{t['repeat']}}</td>
            <td>
              <form action="/task/remove/{{t['id']}}" method="post" style="display:inline">
                <button type="submit">Remove</button>
              </form>
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>

      <h3>Add Scheduled Task</h3>
      <form action="/task/add" method="post">
        <label>Device ID: <input name="device_id" required></label>
        <label>Command: <input name="command" required></label>
        <label>Schedule (ISO datetime for once e.g. 2025-08-10T08:59:00 or HH:MM for daily): <input name="schedule_time" required></label>
        <label>Repeat: <select name="repeat"><option value="once">Once</option><option value="daily">Daily</option></select></label>
        <button type="submit">Add Task</button>
      </form>
    </div>

  </div>
</body>
</html>
"""

@app.route('/')
def index():
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM devices ORDER BY id')
    devices = cur.fetchall()
    cur.execute('''
        SELECT t.*, d.name AS device_name FROM tasks t
        JOIN devices d ON d.id = t.device_id
        ORDER BY t.id
    ''')
    tasks = cur.fetchall()
    conn.close()
    return render_template_string(INDEX_HTML, devices=devices, tasks=tasks)


@app.route('/device/add', methods=['POST'])
def add_device():
    name = request.form.get('name')
    topic = request.form.get('topic')
    conn = db_connection()
    cur = conn.cursor()
    try:
        cur.execute('INSERT INTO devices (name, topic, state) VALUES (?, ?, ?)', (name, topic, 'unknown'))
        conn.commit()
    except sqlite3.IntegrityError:
        # topic must be unique
        pass
    conn.close()
    return redirect(url_for('index'))


@app.route('/device/toggle/<int:device_id>', methods=['POST'])
def device_toggle(device_id):
    cmd = request.form.get('command', 'TOGGLE')
    send_mqtt_command(device_id, cmd)
    return redirect(url_for('index'))


@app.route('/device/command/<int:device_id>', methods=['POST'])
def device_command(device_id):
    cmd = request.form.get('command')
    send_mqtt_command(device_id, cmd)
    return redirect(url_for('index'))


@app.route('/task/add', methods=['POST'])
def add_task():
    device_id = int(request.form.get('device_id'))
    command = request.form.get('command')
    schedule_time = request.form.get('schedule_time')
    repeat = request.form.get('repeat', 'once')
    now = datetime.utcnow().isoformat()
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO tasks (device_id, command, schedule_time, repeat, created_at) VALUES (?, ?, ?, ?, ?)',
                (device_id, command, schedule_time, repeat, now))
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    # schedule it in scheduler
    schedule_task_job(task_id, device_id, command, schedule_time, repeat)
    return redirect(url_for('index'))


@app.route('/task/remove/<int:task_id>', methods=['POST'])
def remove_task(task_id):
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM tasks WHERE id=?', (task_id,))
    conn.commit()
    conn.close()
    # remove scheduled job if exists
    job_id = f"task_{task_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    return redirect(url_for('index'))

# ----------------------- API Endpoints -----------------------

@app.route('/api/devices', methods=['GET'])
def api_devices():
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM devices')
    rows = cur.fetchall()
    conn.close()
    results = [dict(r) for r in rows]
    return jsonify(results)


@app.route('/api/tasks', methods=['GET'])
def api_tasks():
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM tasks')
    rows = cur.fetchall()
    conn.close()
    results = [dict(r) for r in rows]
    return jsonify(results)

# ----------------------- App startup -----------------------

def start_scheduler():
    scheduler.start()
    load_and_schedule_all_tasks()


def start_mqtt():
    setup_mqtt_client()


def ensure_sample_data():
    # Add a sample device if none exist for easy first-run
    conn = db_connection()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) as c FROM devices')
    c = cur.fetchone()['c']
    if c == 0:
        cur.execute('INSERT INTO devices (name, topic, state) VALUES (?, ?, ?)', ('Living Room Light', 'living_room_light', 'off'))
        cur.execute('INSERT INTO devices (name, topic, state) VALUES (?, ?, ?)', ('Fan', 'ceiling_fan', 'off'))
        conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    ensure_sample_data()

    # Start MQTT in background
    start_mqtt()

    # Start Scheduler
    start_scheduler()

    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
