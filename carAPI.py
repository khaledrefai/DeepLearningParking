import os
import tempfile
from datetime import datetime
from os import path

from flask import Flask, jsonify, request, logging,render_template, send_from_directory
from flask import g
import sqlite3

DATABASE = 'ParkingCount.db'
table_name = 'parkingCount'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def Motion():
    return render_template('index.html')


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/images/<path:path>')
def send_gfx(path):
    return send_from_directory('images', path)

@app.route('/js/<path:path>')
def send_t(path):
    return send_from_directory('js', path)

@app.route('/sass/<path:path>')
def send_sass(path):
    return send_from_directory('sass', path)

@app.route('/<path:path>')
def get_file(path):
    return send_from_directory('', path)
@app.route('/getattend')
def getattend():
    with app.app_context():
        c = get_db().cursor()
        c.execute("SELECT * FROM " + table_name)
        rows = c.fetchall()
        return jsonify(rows)


@app.route('/getlastattend')
def getlastattend():
    with app.app_context():
        c = get_db().cursor()
        c.execute("SELECT parkingNumber ,carCount FROM " + table_name + " order by logtime desc limit 1 ")
        row = c.fetchone()
        return jsonify(row)

def get_db():
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE)
        return db


@app.teardown_appcontext
def close_connection(exception):
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "not ok", "message": "this server could not understand your request"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "message": "route not found"})


@app.errorhandler(500)
def not_found(e):
    return jsonify({"status": "internal error", "message": "internal error occurred in server"})


@app.route('/detect', methods=['POST'])
def postimage():
    file = request.files.get('upload')
    filename, ext = os.path.splitext(file.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
         return 'File extension not allowed.'
        # loading the trained weights
    tmp = tempfile.TemporaryDirectory()
    temp_storage = path.join(tmp.name, file.filename)
    file.save(temp_storage)
    timess = str(datetime.datetime.now())
    totalParking=0
    totalCar=0
    print('time: ',timess)
    with app.app_context():
        c = get_db().cursor()
        c.execute("INSERT INTO " + table_name + " VALUES (" + totalParking + ", '" + totalCar + ", '" + timess + "')")
        get_db().commit()
    return jsonify(timess)



if __name__ == '__main__':
    with app.app_context():
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        c = get_db().cursor()
        sql = 'create table if not exists ' + table_name + ' (parkingNumber integer,carCount integer , logtime text)'
        c.execute(sql)
        get_db().commit()
        c.close()
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("Starting server on http://localhost:5000")
        print("Serving ...", app.run(host='0.0.0.0', port=5000))
        print("Finished !")
        print("Done !")
