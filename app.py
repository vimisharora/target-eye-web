from flask import Flask, render_template, redirect, url_for
# from camera import *

app=Flask(__name__)

@app.route('/')
def welcome():
    import camera as cam
    # render_template("welcome.html")
    cam
    return redirect(url_for('index'), code=302)


@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/entertainment')
def entertainment():
    return render_template("Entertainment.html")

@app.route('/food')
def food():
    return render_template("food.html")

@app.route('/issues')
def issues():
    return render_template("issues.html")

@app.route('/keyboard')
def keyboard():
    return render_template("keyboard.html")

@app.route('/medicine')
def medicine():
    return render_template("medicine.html")

@app.route('/normal')
def normal():
    return render_template("normal.html")

@app.route('/quick_message')
def quick_message():
    return render_template("quick_message.html")

@app.route('/water')
def water():
    return render_template("water.html")

app.run()